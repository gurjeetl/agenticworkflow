# Stakeholder Summary — Agent Framework Progress

**As of:** 2026-05-31
**Scope:** progress against the reference architecture (Planner → Orchestrator → Gate → Synthesizer with supporting subsystems)
**Headline:** 11 of 14 architecture items are in production-ready shape. 2 remain (vector-based agent discovery, Kafka transport) and are the focus of the next iteration. 1 is intentionally deferred.

---

## What's completed

1. **Planner** — turns a single user request into a plan of subtasks, picks the right agent for each, and wires up dependencies. Supports re-planning when the first attempt is incomplete.
2. **Orchestrator** — runs the plan in dependency waves, fanning out independent tasks in parallel and waiting only where one task needs another's output. Retries failed tasks once.
3. **Wave execution** — confirmed parallel execution within each wave; sequential only where the graph demands it.
4. **Shared blackboard** — every agent writes to and reads from a common workspace. Backed by Redis when available, in-memory otherwise. Lets us audit and replay any run.
5. **Completion gate** — decides "are we done?" after each run; re-plans up to 3 times if not, then forces a partial response.
6. **Synthesizer** — composes the final answer from the blackboard, marking sections `[PARTIAL]` where agents errored.
7. **Agent registry (schema)** — every agent declares its capability, inputs, outputs, version, SLA, and status. Registry is the single source of truth.
8. **Memory & state (3 of 4 stores)** — Redis (hot, per-run), Postgres (durable audit + facts), MongoDB (conversation history). All three degrade gracefully when offline.
9. **Observability** — end-to-end tracing via MLflow. Every run is identified by a `run_id`; every planner, agent, tool, and synthesizer call shows up as a span with version + timing + retry counts.
10. **Agent versioning** — semantic versioning on every agent, pinned at plan time, committed alongside outputs. Any run is reproducible from its `run_id`.
11. **Tool / MCP gateway (basic)** — agents can call external tools through a controlled gateway with permission filtering. Spans emitted to the trace.

---

## What's pending — in scope for next iteration

12. **Vector-based agent discovery (Milvus)** — today the Planner sends every agent's description into the LLM prompt. This works for ~2 agents but will break at 100+. The next iteration embeds each agent profile into Milvus and only hands the Planner the top 5–7 most relevant agents per request. *Unlocks scale.*
13. **Kafka A2A transport** — agents currently run in-process. Adding Kafka as an async transport, with a selector that picks sync vs async per task, lets long-running or fan-out agents run independently without blocking the request. *Unlocks resilience and out-of-process workers.*
14. **Kafka backpressure** — comes with item 13: monitors queue lag, pauses publishing when consumers fall behind, sends repeatedly failing messages to a dead-letter queue.

---

## What's pending — deferred for now

- **Partial streaming to the user** — the platform already emits wave-done events internally; surfacing them as a live stream over the API is a small, separable change. Deferred until the two items above are in.
- **Milvus for semantic memory** — the same Milvus instance that backs agent discovery will later store outage/RAG embeddings for cross-run lookups.
- **Per-tool timeout + token budget** — tool calls currently honour the agent's SLA but not per-tool bounds. Cheap follow-up.
- **OpenTelemetry exporter** — MLflow covers our tracing needs today. Adding an OTel exporter (Jaeger / Tempo / Honeycomb) is additive, not urgent.

---

## Risks worth flagging

- **Registry scale ceiling.** Today's "send all agents to the LLM" approach is the single biggest scale risk. Vector discovery (item 12) directly removes it; until then, we should cap the registry at ~15 agents.
- **In-process dispatch.** A slow or crashing agent today can stall the request. Kafka transport (item 13) isolates this.
- **No partial response over the wire yet.** If a request takes >10s, the user sees nothing until it finishes. Acceptable for now given typical request times; revisit once partial streaming lands.

---

## One-line status

> Core pipeline is built and observable. Next iteration removes the two scale-limiting gaps — agent discovery and async transport — so we can grow past a handful of agents and tolerate slow ones without blocking users.
