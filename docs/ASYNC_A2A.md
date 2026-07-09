# Asynchronous A2A Communication over Kafka

> The single reference for the async agent-communication subsystem: what was built, how it works,
> how to run and test it, and what remains open.
> Visual companion: [a2a-bus-architecture.html](a2a-bus-architecture.html) (sequence diagram,
> control plane, security gap, multi-team target).

## 1. Why it exists

The original A2A layer was fully synchronous: the Executor blocked a worker thread on JSON-RPC
`message/send` until every agent in a wave replied. That cannot express what an enterprise system
needs: buffering when an agent is busy or down, deadlines with escalation instead of hangs,
retries without double-processing, dead-lettering with replay, and out-of-band control (cancel,
re-plan) while work is in flight.

The async design keeps the graph and the message shapes, and adds **Kafka as the transport** with
**durable suspend/resume** as the execution model: an agent call becomes
*produce → durable suspend → reply → resume*, surviving process restarts in between.

Everything is **opt-in**: with `KAFKA_ENABLED=false` (the default) the platform behaves exactly as
before, and async is enabled per agent via `AgentMeta.transport` (`"json-rpc"` | `"kafka"` | `"both"`).

## 2. Topic taxonomy & message contract

| Topic | Purpose | Key | Consumers |
|---|---|---|---|
| `genie.agents.<agent_id>.inbox` | requests to one agent | `thread_id` (per-conversation ordering) | that agent's consumer group |
| `genie.replies` | all replies **and** control msgs (`step.cancelled`) | `thread_id` | gateway reply-router group |
| `genie.dlq` | dead letters (poison pills + retries exhausted), replayable | `thread_id` | Supervisor group + operators |

One **shared reply topic** matched by correlation id (per-run topics hit Kafka's cardinality
limits; Redis pub/sub would lose durability/replay). Topic names derive from `BUS_TOPIC_PREFIX`
(default `genie`).

- **Kafka value** = the a2a-sdk `Message`/`Task` JSON — identical to the synchronous HTTP wire shape.
- **Kafka headers** = routing/control, so no router ever deserializes a body: `correlation_id`,
  `attempt`, `kind` (`request|reply|step.cancelled|dead_letter`), `from`, `to`, `reply_to`,
  `thread_id`, `run_id`, `task_id`, `deadline` (RFC3339), `group_id`, `trace_id`, `tenant_id`.
- **Deterministic correlation id**: `uuid5(ns, "{run_id}:{task_id}:{attempt}")`. LangGraph
  re-executes an interrupted node from the top on resume, so the dispatch re-runs — a
  deterministic cid turns the re-produce into a harmless duplicate instead of a second execution.

### Dedup (Kafka is at-least-once) — three claim domains, all Redis `SETNX`

| Claim | Guarantees |
|---|---|
| `dedup:inbox:{agent}:{cid}:{attempt}` | a redelivered request never re-runs the agent; a retry (`attempt+1`) passes |
| `dedup:reply:{cid}` | exactly one resolution per wait — first of real reply / timeout / cancel wins, late arrivals ignored |
| `dedup:resume:{group_id}` | exactly one graph resume per dispatch group, even across gateway instances |

Redis is **required** in async mode (startup fails fast without it); still optional for sync mode.

## 3. End-to-end flow

```
Executor                    Kafka                      Agent B                Gateway
   |                          |                           |                      |
   | 1 write a2a_awaiting     |                           |                      |
   | 2 produce ------------->  inbox topic --------------> 3 dedup(cid,attempt)  |
   | 3 interrupt() ═ suspend  |                           |   _run_task()        |
   |   (durable checkpoint)   |                           | 4 produce reply ---> genie.replies
   |                          |                           |   (poison → genie.dlq)
   |                          |                                                  | 5 match cid → awaiting
   |                          |                                                  |   claim dedup:reply:<cid>
   | <------ 6 graph.invoke(Command(resume=results)) in background task --------|
   | 7 blackboard ← results; next wave / gate / synthesizer …                    |
```

- **HTTP contract**: `POST /chat` returns **202 `{status:"pending", thread_id, run_id}`** when the
  run suspends on a bus task; `GET /runs/{thread_id}/{run_id}` reads the durable checkpoint
  (`pending` | `completed` + response | `unknown`). Sync-only runs return inline exactly as before.
  The frontend polls `/runs/...` on a 202 and renders the answer when it lands.
- **Race safety**: the awaiting record is written **before** the produce, so a fast reply can never
  race an unrecorded wait (early replies are parked and retried briefly); the reply-dedup claim
  prevents a real reply and the timeout sweep from both resolving the same wait.
- **No suspend-forever**: every wait carries a deadline. With the Supervisor disabled, the gateway
  reply-router sweeps expired waits itself; an expired wait resumes the run with a deadline error →
  blackboard error → the existing Gate → Planner re-plan / partial synthesis.
- **Poison pills**: a payload that fails schema validation goes straight to the DLQ with the parse
  error in headers and the original payload preserved — no reply, no retry (every retry would fail
  identically). A *business* failure (agent ran, returned an error) is **not** dead-lettered — it
  returns as a `failed` Task reply, identical to the sync contract.

### Executor design — wave-per-invocation

`interrupt()` re-executes the whole node, so the Executor runs **one wave per invocation**, driven
by a `wave_cursor` state field, with a conditional edge looping `executor → executor` until waves
are exhausted, then `→ gate`. Each finished wave commits to the checkpoint, so completed waves
survive resumes and restarts. Within a wave, **bus tasks all produce up front** and the run
suspends **once** for the whole group (`group_id = run:wave`); when the last record of the group
resolves (reply / timeout / cancel), the run resumes with the combined payload
`{task_id: {"task": ...} | {"error": ...}}`. Sync tasks run after the bus dispatch so they execute
exactly once. The Orchestrator resets `wave_cursor=0` on every (re)plan.

**`call_peer` stays synchronous.** An agent is not a graph and cannot suspend — `BaseAgent.call_peer`
always uses HTTP JSON-RPC regardless of the peer's transport. Only the Executor uses the bus, via
`A2AClient.send_via_bus(...) -> correlation_id`.

### Fast path with bus fallback — `transport="both"`

The Executor tries the direct HTTP call first; on failure it re-enters the same wave with the task
forced onto the bus (`bus_fallback` state field), which then suspends/resumes like any bus task.
No double-execution: the fallback re-entry happens in a fresh executor invocation, and a
bus-failed task never re-falls-back.

## 4. Control plane — the A2A Supervisor

A standalone process (`services/supervisor/server.py`, logic in
`src/genie/messaging/supervisor.py`). Run it and set `BUS_SUPERVISOR_ENABLED=true` on the gateway;
the gateway's simple timeout sweep then stands down. Per expired wait, the failure ladder:

| Condition | Action |
|---|---|
| extensions left **and** agent heartbeat healthy in the registry | **EXTEND** the deadline (`bus_max_extends`, default 3) — "slow, not dead" |
| attempts left (`bus_max_attempts`, default 2) | **RETRY**: re-produce the stored request as `attempt+1` (new deterministic cid, same group; inbox dedup absorbs double-produces) |
| exhausted | **DEAD-LETTER** to `genie.dlq` with full payload + error headers, replayable |

The Supervisor also **permanently consumes `genie.dlq`** (its own letters *and* agents' poison
pills) and produces **`step.cancelled`** to `genie.replies` — the reply-router converts the wait
into a cancelled result and the run unblocks immediately instead of burning its deadline.
`send_via_bus` stores each request's payload + inbox topic on the awaiting record precisely so the
Supervisor can retry/dead-letter without re-reading Kafka. The Supervisor deliberately has no
graph: all unblocking flows through the bus (control plane), never through tracing.

**Operator tools:**

- `POST /runs/{thread_id}/{run_id}/cancel` — resolves every pending wait of the run as cancelled
  and group-resumes; the run finishes with a partial answer (409 when async mode is off).
- DLQ replay CLI:
  ```powershell
  uv run python scripts/replay_dlq.py list
  uv run python scripts/replay_dlq.py replay --cid <correlation_id>   # or --all [--dry-run]
  ```

## 5. Components

| Component | Location | Role |
|---|---|---|
| `Broker` protocol + `KafkaBroker` + `FakeBroker` | `src/genie/messaging/broker.py` | produce/consume abstraction; Fake runs the test suite without Docker |
| Envelope helpers | `src/genie/messaging/envelope.py` | headers, topic names, deterministic cid |
| Dedup | `src/genie/messaging/dedup.py` | Redis SETNX claims (three domains, §2) |
| Awaiting store | `src/genie/messaging/awaiting.py` | Mongo `a2a_awaiting`: who waits for which cid, deadline, status, stored request; TTL cleanup |
| Durable checkpointer | `src/genie/application/checkpointer.py` | `MongoDBSaver` when async is on; `MemorySaver` otherwise |
| Agent bus consumer | `src/genie/agents/server.py` | inbox → dedup → `_run_task` (same code path as HTTP) → reply; poison → DLQ |
| Bus send | `src/genie/a2a/client.py` | `send_via_bus`; HTTP `send()` untouched |
| Wave-cursor Executor | `src/genie/application/nodes/executor.py` | suspend/resume per §3 |
| ReplyRouter (+ sweep) | `src/genie/interface/reply_router.py` | resume runs; deadline sweep when Supervisor is off |
| Pending-run API | `src/genie/interface/routers/runs.py` | `GET /runs/...` + cancel |
| A2A Supervisor | `src/genie/messaging/supervisor.py` · `services/supervisor/` | failure ladder + DLQ consumer (§4) |
| Local infra | `docker-compose.yml` | Redpanda :9092, Redis :6379, Mongo :27017 |

*(Naming: the control-plane box ships as the **Supervisor** to avoid colliding with the graph's
`orchestrator.py` decomposition node.)*

## 6. Enterprise features

- **Multi-tenancy**: optional `ChatRequest.tenant_id` flows through state → bus headers →
  awaiting records → blackboard keys (`bb:{tenant}:{thread}:{run}:{task}`). Unset = single-tenant
  with pre-tenancy key shapes preserved byte-for-byte.
- **Blackboard read-through**: `Blackboard.get` falls back to the Redis mirror on a local miss, so
  a process that didn't produce an entry (another gateway instance) resolves it when ready.
  Entries are keyed by (tenant, thread, **run**, task) and run ids are uuid4-unique, so stale
  cross-run reads cannot occur.
- **Trace propagation**: every bus hop carries a `trace_id` header (currently = `run_id`), and
  every consumer/router/Supervisor decision logs with those attributes. The observability plane
  sees each hop; the control plane never depends on it.

## 7. Configuration

All env-first via `genie.platform.config.Settings` (secrets → `.env` / `config/local.yaml`, never
`config/default.yaml`):

| Setting | Default | Purpose |
|---|---|---|
| `kafka_enabled` | `false` | Master switch for the async transport |
| `kafka_bootstrap_servers` | `localhost:9092` | Broker list |
| `kafka_security_protocol` | `PLAINTEXT` | `SSL` / `SASL_PLAINTEXT` / `SASL_SSL` |
| `kafka_sasl_mechanism` / `_username` / `_password` | unset | SASL auth (`PLAIN`, `SCRAM-SHA-256/512`) |
| `kafka_ssl_cafile` | unset | Org CA bundle |
| `bus_topic_prefix` | `genie` | Org naming conventions (e.g. `oati.genie`) |
| `bus_reply_topic` / `bus_dlq_topic` | derived | `{prefix}.replies` / `{prefix}.dlq` |
| `bus_consumer_group` | `genie-gateway` | Reply-router consumer group |
| `bus_dedup_ttl_seconds` | `900` | Dedup-claim TTL (≥ longest deadline) |
| `a2a_default_deadline_ms` | `60000` | Reply deadline for a bus task |
| `bus_sweep_interval_seconds` | `5.0` | Deadline-sweep cadence |
| `bus_supervisor_enabled` | `false` | Gateway sweep stands down; Supervisor owns the ladder |
| `bus_max_extends` / `bus_max_attempts` | `3` / `2` | Supervisor ladder budgets |

| Mode | Mongo | Redis | Kafka |
|---|---|---|---|
| Sync (default) | required | optional | — |
| Async (`KAFKA_ENABLED=true`) | required (checkpoints + awaiting) | **required** (dedup — fail fast) | required |

## 8. Running it

### Configuring an agent for async communication

Async is a **declaration, not an implementation**: the agent's business code (`agent.run`) is
untouched — bus-delivered requests go through the exact same `_run_task` path as HTTP ones. An
agent opts in through its `AgentMeta` record:

```python
META = AgentMeta(
    agent_id="weather",
    capability_tags=["weather", "forecast", "city"],
    description="Reports current weather conditions for a named city.",
    transport="kafka",        # ← the only required change (default is "json-rpc")
    # inbox_topic="oati.genie.agents.weather.inbox",   # optional override; None derives
    #                                                  # "{bus_topic_prefix}.agents.{agent_id}.inbox"
)
```

What each `transport` value means:

| Value | Behavior |
|---|---|
| `"json-rpc"` (default) | synchronous HTTP only — pre-async behavior, byte for byte |
| `"kafka"` | Executor dispatches via the bus; the run suspends and resumes on the reply |
| `"both"` | fast path: direct HTTP first, automatic bus fallback on failure (§3) |

Prerequisites on the platform side (not per agent):

1. `KAFKA_ENABLED=true` and `REDIS_URL` set (`.env` / `config/local.yaml`) — the agent harness
   starts its inbox consumer automatically when both the flag and a kafka-capable transport are
   present, and **fails fast at startup** if Redis is missing.
2. The inbox topic must exist — auto-created in local dev; pre-created by the platform team on
   corporate clusters (see the checklist below).
3. Nothing changes for peer calls: `call_peer` stays synchronous HTTP regardless of transport.

Rollback is the same one line: set `transport="json-rpc"` back and the agent is fully synchronous
again — no code or data migration involved.

### Local development (Redpanda via Docker)

```powershell
docker compose up -d                       # Redpanda :9092, Redis :6379, Mongo :27017
# .env: KAFKA_ENABLED=true / REDIS_URL=redis://localhost:6379
# opt an agent in: AgentMeta(transport="kafka")   # or "both" for the fast path
powershell -ExecutionPolicy Bypass -File scripts\run-full.ps1
# optional control plane:
# BUS_SUPERVISOR_ENABLED=true + uv run python -m services.supervisor.server
```

Chat normally at `http://localhost:8000` — a prompt routed to a kafka-transport agent goes 202 →
poll → answer; everything else behaves as before.

### Corporate Apache Kafka — config only, zero code changes

`aiokafka` speaks the Kafka wire protocol; Redpanda is only the local stand-in.

1. `KAFKA_BOOTSTRAP_SERVERS=broker1:9093,broker2:9093`, security settings per §7.
2. **Pre-create topics** (corporate clusters disable auto-create):
   `{prefix}.agents.<id>.inbox` (3–6 partitions, 1–7 d retention) per agent ·
   `{prefix}.replies` (6+, 1–7 d) · `{prefix}.dlq` (3, **30+ d** — must stay replayable).
3. **ACLs per principal**: agents → read own inbox, write `replies`/`dlq`; gateway → write
   inboxes, read `replies`; Supervisor → read `dlq`, write inboxes + `replies`.
4. Monitor consumer lag on the three groups + DLQ depth in the org's existing tooling.

## 9. Testing

Automated — the whole suite runs on `FakeBroker`, **no Docker needed**:

```powershell
uv run pytest                              # full suite
uv run pytest tests/unit/test_messaging_envelope.py tests/unit/test_messaging_dedup.py `
              tests/unit/test_supervisor.py `
              tests/integration/test_agent_bus_consumer.py `
              tests/integration/test_async_executor.py `
              tests/integration/test_reply_router.py -q      # just the async A2A suites
uv run pytest -m kafka -q                  # real-broker e2e; needs KAFKA_BOOTSTRAP_SERVERS set
```

Manual walkthrough (async mode on, one agent on `transport="kafka"`):

1. **Happy path** — ask something routed to the bus agent; watch `chat.pending` →
   `a2a.bus.replied` → `a2a.reply_router.run_completed`; `/chat` returns 202 and
   `GET /runs/...` flips `pending → completed`.
2. **Durability** — send a prompt, kill the gateway before the reply, restart: the run completes
   from the Mongo checkpoint.
3. **Poison pill** — `rpk topic produce genie.agents.<id>.inbox`, type garbage; the dead letter in
   `genie.dlq` carries `schema_validation_failed` with the payload preserved.
4. **Ladder** — stop the agent, send a prompt: `extend_granted` → `retried` → `dead_lettered` →
   `step_cancelled` → run completes with a partial answer; `replay_dlq.py list` shows the payload.

## 10. Guarantees

- **Backward compatible** — default config = old behavior; async is opt-in per agent.
- **No suspend-forever** — every wait has a deadline (gateway sweep or Supervisor ladder).
- **No double-execution** — at-least-once delivery + deterministic cids + three-domain dedup.
- **Restart-safe** — suspended runs live in Mongo, resumable by any gateway instance.
- **Exactly-one resume per wave** — group-level Redis claim, even across instances.

## 11. Remaining gaps (verified against the code)

- **No push notifications** — `AgentCapabilities.push_notifications` is hardcoded `False`
  (`src/genie/a2a/agent_card.py`); no webhook delivery for long-running work.
- **No file parts** — only `TextPart`/`DataPart` cross the A2A boundary.
- **Single Registry** — TTL-based liveness, stale-on-failure fallback, but no federation story.
- **Trust boundary is by convention** — all bus clients share one credential today; per-principal
  ACLs and per-team topic namespaces are designed (see
  [a2a-bus-architecture.html](a2a-bus-architecture.html) §3–4) but not automated.
- **Ops tooling** — consumer-lag dashboards / DLQ alerting live in org tooling, not the platform.
- **Late attempt-1 replies** after a retry was dispatched are dropped by design (the retry wins).

---

*This document supersedes `A2A_GAPS.md`, `ASYNC_A2A_PLAN.md` and `ASYNC_A2A_PHASE1.md` (deleted);
closed gaps from those docs — synchronous-only transport, no Task lifecycle, fixed single retry,
no delivery guarantees, no streaming — are simply the features described above.*
