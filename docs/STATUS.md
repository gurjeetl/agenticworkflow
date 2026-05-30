# Architecture Status — done vs pending

This document maps the current state of `BaseAgentFramework` against the 14 items in our reference flow diagram (Planner → Orchestrator → Gate → Synthesizer, plus the supporting subsystems: blackboard, registry, memory, A2A transport, observability, versioning, partial streaming, Kafka backpressure).

Last updated: 2026-05-31.

## Summary table

| # | Diagram item | Status | Primary file(s) | Gap |
|---|---|---|---|---|
| 1 | Planner | Done | [planner/planner_agent.py](../planner/planner_agent.py) | None |
| 2 | Orchestrator | Done | [orchestrator/orchestrator.py](../orchestrator/orchestrator.py) | Transport selector + Kafka lag check |
| 3 | Wave execution | Done | [orchestrator/orchestrator.py:64](../orchestrator/orchestrator.py#L64) | None |
| 4 | Tool / MCP gateway | Partial | [baseagent/mcp_client.py](../baseagent/mcp_client.py), [baseagent/permissions.py](../baseagent/permissions.py) | Per-tool timeout + token budget bounds |
| 5 | Blackboard | Done | [orchestrator/blackboard.py](../orchestrator/blackboard.py) | Wave-done SSE flush (partial streaming) |
| 6 | Completion gate | Done | [gate/completion_gate.py](../gate/completion_gate.py) | None |
| 7 | Synthesizer | Done | [synthesizer/synthesizer.py](../synthesizer/synthesizer.py) | None |
| 8 | Agent registry | Done (schema) / **Pending discovery** | [registry/registry.py](../registry/registry.py), [registry/agent_meta.py](../registry/agent_meta.py) | **No vector discovery — full menu sent to LLM** |
| 9 | Memory & state | Done (Redis, Postgres, Mongo) / **Pending Milvus** | [memory/](../memory/) | Milvus for semantic memory + registry embedding |
| 10 | A2A transport | Partial (in-process only) | `asyncio.to_thread` in orchestrator | **JSON-RPC + Kafka transports + selector** |
| 11 | Observability | Done (MLflow) | [observability/](../observability/) | OpenTelemetry exporter (optional, deferred) |
| 12 | Agent versioning | Done | [registry/agent_meta.py:24](../registry/agent_meta.py#L24) | None |
| 13 | Partial streaming | Pending | — | Deferred |
| 14 | Kafka backpressure | Pending | — | Comes with item 10 |

Overall: **11 of 14 fully done**, 2 partial, 1 pending. Two strategic gaps remain: **vector-based agent discovery** (registry/Planner) and **Kafka A2A transport** (orchestrator).

---

## What's done — in detail

**1. Planner** — [planner/planner_agent.py](../planner/planner_agent.py). Splits the user prompt into a DAG of `Subtask`s, normalizes LLM tics on `agent_id`, validates args against each agent's `input_schema`, and pins `agent_version` at plan time. Supports re-entry: when the Gate triggers a re-plan, the previous `blackboard_snapshot` + `replan_reason` are folded back into the prompt (lines 58–67).

**2. Orchestrator** — [orchestrator/orchestrator.py](../orchestrator/orchestrator.py). Executes the DAG wave-by-wave using `Plan.waves()` (Kahn's algorithm in [planner/dag.py](../planner/dag.py)). Each wave fans out via `asyncio.gather` (line 66), skips tasks already satisfied by a snapshot (line 56), retries once per task, and writes results / errors to the blackboard.

**3. Wave execution** — `_run_wave` ([orchestrator/orchestrator.py:64](../orchestrator/orchestrator.py#L64)) emits `orchestrator.wave_start` / `orchestrator.wave_done` events with the wave index.

**5. Blackboard** — [orchestrator/blackboard.py](../orchestrator/blackboard.py). Two-layer: in-process dict mirrored to Redis when `REDIS_URL` is set. Keys: `bb:{thread_id}:{run_id}:{task_id}`. 24h TTL. Graceful no-op when Redis is unavailable.

**6. Completion gate** — [gate/completion_gate.py](../gate/completion_gate.py). Checks all subtasks present, error flags, and the re-plan budget (`max_replans`, default 3). Sets `state["next_action"]` to `"synthesize"` or `"replan"`.

**7. Synthesizer** — [synthesizer/synthesizer.py](../synthesizer/synthesizer.py). Reads the blackboard and composes one answer along three paths: empty plan → clarification; single task with a structured view → pass-through; otherwise → LLM-compose prose. Marks `[PARTIAL]` for error entries. Commits `persist=true` fields to Postgres.

**8. Agent registry — schema** — [registry/agent_meta.py](../registry/agent_meta.py). `AgentMeta` records `agent_id`, `version` (semver), `capability_tags`, `description`, `input_schema`, `output_schema` (per-field `persist` flag), `sla_ms`, `transport`, `status`, `changelog_url`. Required-field validation in `validate_args`.

**9. Memory & state — Redis / Postgres / Mongo** — [memory/](../memory/). Redis ([memory/redis_store.py](../memory/redis_store.py)) is the blackboard hot store. Postgres ([memory/postgres_store.py](../memory/postgres_store.py)) auto-creates `agent_commits` + `entity_links` and is the commit target for `persist=true` outputs. Mongo ([memory/mongo_store.py](../memory/mongo_store.py)) handles conversation messages + thread facts. All three degrade gracefully when their env vars are unset.

**11. Observability — MLflow** — [observability/](../observability/). Every component inheriting `Observable` ([observability/observable.py](../observability/observable.py)) gets its traced methods wrapped in MLflow spans. Per-agent spans carry `agent.id`, `agent.version`, `orch.wave`, `task.id`, `retry.count`. Structured JSON logging routes log records as span events. `run_id` is generated per request in [main.py](../main.py) and threaded through state.

**12. Agent versioning** — [registry/agent_meta.py:24](../registry/agent_meta.py#L24). Semver on every agent; the Planner pins it into each `Subtask.agent_version` and the Synthesizer commits it alongside output rows in Postgres, so any run is reproducible from `run_id`.

---

## What's partial

**4. Tool / MCP gateway** — [baseagent/mcp_client.py](../baseagent/mcp_client.py) loads MCP tools via `MultiServerMCPClient` and [baseagent/permissions.py](../baseagent/permissions.py) filters them by permission. MLflow spans capture tool calls. **Missing:** per-tool timeout enforcement and a per-call token budget — today an agent can run a slow tool to its natural conclusion regardless of the agent's `sla_ms`.

**10. A2A transport** — Dispatch today is in-process: `asyncio.to_thread(agent.run, task_state)` in [orchestrator/orchestrator.py:88](../orchestrator/orchestrator.py#L88). The `AgentMeta.transport` field (`"json-rpc" | "kafka" | "both"`, default `"json-rpc"`) exists but is dormant — the Orchestrator does not read it. There is no `Transport` abstraction, no selector, no Kafka producer/consumer. This is the largest gap.

---

## What's pending — in scope for the next iteration

### A. Vector-based registry discovery (Milvus)
The Planner currently calls `list_active()` and renders **every** active agent into one LLM system prompt ([planner/planner_agent.py:38–54](../planner/planner_agent.py#L38-L54)). This is fine for 2 agents but breaks at 100+. The next iteration introduces a two-tier semantic router (per `registry.txt`):

- Embed each agent profile (description + tags + input descriptions) on registration.
- Store embeddings in Milvus alongside `agent_id`, `version`, `embedding_model`.
- At plan time, embed the user prompt and vector-search for top-K (default 7) agents.
- Render only the shortlisted agents into the Planner prompt.
- Graceful fallback to `list_active()` when Milvus is unset or returns nothing.

Touches: [registry/registry.py](../registry/registry.py), [registry/agent_meta.py](../registry/agent_meta.py), [planner/planner_agent.py](../planner/planner_agent.py); adds `registry/embedding_index.py` and `registry/discovery.py`.

### B. Kafka A2A transport + transport selector
Introduce a `Transport` protocol with two implementations — `JsonRpcTransport` (wraps today's in-process call) and `KafkaTransport` (aiokafka producer/consumer with correlation_id reply pattern, lag-based backpressure, and DLQ on max retry). A `TransportSelector` picks per task: blocking dependency → JSON-RPC, otherwise honour `meta.transport`.

Touches: [orchestrator/orchestrator.py](../orchestrator/orchestrator.py); adds `transport/` package.

### C. Kafka backpressure (item 14)
Implemented inside `KafkaTransport.dispatch`: the Orchestrator checks consumer lag before publishing the next wave; if `lag > KAFKA_LAG_THRESHOLD` it backs off exponentially. Messages exceeding `KAFKA_MAX_RETRIES` go to `agents.{agent_id}.dlq`.

---

## Explicitly deferred

- **13. Partial streaming (wave-done SSE flush).** The blackboard already publishes wave-done events; surfacing them as SSE on `POST /chat` is a small change but separable from the discovery/transport work and is not in scope this iteration.
- **OpenTelemetry exporter.** MLflow Tracing covers our observability needs end-to-end today. A future OTel exporter would let us forward spans to Tempo / Jaeger / Honeycomb, but it's an additive concern, not a gap.
- **Per-tool timeout + token budget (item 4 gap).** Easy to add once the broader bounds-policy interface (Orchestrator timeout enforcement) lands; tracked separately.

---

## API surface today

Three public routes in [main.py](../main.py):

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | Run the full graph (Planner → Orchestrator → Gate → Synthesizer) and return a final answer + optional structured view. |
| GET | `/health` | Liveness check. |
| GET | `/state/{thread_id}` | Return the last LangGraph checkpoint for a thread (used by the trace UI). |

Plus one internal route — `POST /chat/trace` — that runs the same graph but emits per-node step events for the animated trace UI.

---

## Required environment variables

- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MCP_AUTH_TOKEN` — no defaults; the app fails fast without them.
- Optional but recommended: `REDIS_URL`, `POSTGRES_DSN`, `MONGODB_URI`, `MLFLOW_TRACKING_URI`.
- Coming with the next iteration: `MILVUS_URI`, `MILVUS_TOKEN`, `MILVUS_COLLECTION`, `EMBEDDING_MODEL`, `PLANNER_TOPK`, `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_TOPIC_PREFIX`, `KAFKA_CONSUMER_GROUP`, `KAFKA_LAG_THRESHOLD`, `KAFKA_MAX_RETRIES`.
