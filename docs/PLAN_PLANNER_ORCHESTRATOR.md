# Plan: Implement Planner → Orchestrator → Synthesizer architecture

## Context

The flow diagram (`Flow_Diagram_Memory_Async.html`) describes an agentic platform with a Planner that splits a prompt into a dependency-graph (DAG) of agent tasks, an Orchestrator that runs those tasks in dependency waves, a shared blackboard, a completion gate, and a Synthesizer that composes the final answer. Today this repo is much simpler: [main.py](../main.py) calls a LangGraph topology built in [graph/graph_builder.py](../graph/graph_builder.py) — `START → RouterAgent → (WeatherAgent | OutageAgent) → END`. A single leaf agent answers each request; there is no plan, no DAG, no parallel fan-out, no blackboard, and no synthesizer.

This plan brings the diagram's core architecture in, while keeping LangGraph as the runtime shell and reusing every existing primitive ([BaseAgent](../baseagent/base_agent.py), [Observable](../observability/observable.py), [MCPClient](../baseagent/mcp_client.py), [AgentMemory](../baseagent/agent_memory.py)) as-is.

**Scope this iteration:** Core + memory backends — Planner, Orchestrator with DAG/waves, blackboard, completion gate, Synthesizer, agent registry, plus Redis (hot blackboard) and Postgres (durable commit). Defer Kafka A2A transport, Milvus semantic memory, partial streaming, and Kafka backpressure to a later phase.

**Runtime choice:** Keep LangGraph. The outer graph becomes `Planner → Orchestrator → Gate → (Synthesizer | back to Planner)`. The DAG itself executes *inside* the Orchestrator node using `asyncio.gather` — no per-request dynamic graph mutation, no nodes per task.

---

## Outer LangGraph topology (replaces current router-based flow)

```
START
  → planner          (LLM: prompt + registry → DAG)
  → orchestrator     (executes DAG waves in-process; writes to blackboard)
  → gate             (checks: all tasks done? errors? iter ≤ max_replans?)
      ├─ done    → synthesizer → END
      └─ re-plan → planner    (with current blackboard snapshot)
```

`RouterAgent` is removed from the graph. `WeatherAgent` and `OutageAgent` remain as leaf agents, re-registered via the new registry. Their `run()` bodies are unchanged.

---

## New components

### 1. Agent registry — `registry/`

- `registry/agent_meta.py` — Pydantic `AgentMeta`: `agent_id, version, capability_tags, input_schema, output_schema (with per-field persist flag), sla_ms, transport, status`.
- `registry/registry.py` — process-wide dict. `register()`, `get()`, `list_active()`, `match_by_capability()`.
- Each agent file declares a module-level `META = AgentMeta(...)` and calls `register(META, cls)` at import.

### 2. Planner — `planner/`

- `planner/dag.py` — `Subtask` and `Plan` with `waves()` (Kahn's algorithm). Cycle check raises.
- `planner/planner_agent.py` — `PlannerAgent(BaseAgent)` with `tool_names=[]`. Renders the registry into its system prompt; emits a JSON DAG; validates against registry; writes `state["plan"]`.
- Re-plan path: planner sees `state["blackboard_snapshot"]` and `state["replan_reason"]`.

### 3. Orchestrator — `orchestrator/`

- `orchestrator/blackboard.py` — `Blackboard` wraps (a) in-memory `state["blackboard"]` and (b) Redis at `bb:{thread_id}:{run_id}:{task_id}` with TTL.
- `orchestrator/orchestrator.py` — `Orchestrator(Observable)`. Loads plan → computes waves → per-wave `asyncio.gather` → narrowed-state agent dispatch with retry → writes to blackboard.

### 4. Completion gate — `gate/completion_gate.py`

- `CompletionGate(Observable)`. Inspects blackboard, sets `state["next_action"] = "synthesize" | "replan"`. Enforces `max_replans` (default 3).

### 5. Synthesizer — `synthesizer/`

- `SynthesizerAgent(BaseAgent)` with `tool_names=[]`. Reads blackboard, composes one answer marking `[PARTIAL]` for error entries. Uses `set_final_view()` so the existing `/chat` response shape is preserved. Commits `output_schema.persist=true` fields to Postgres.

### 6. Memory backends — `memory/`

- `memory/redis_store.py` — `redis.asyncio` wrapper. Env: `REDIS_URL`.
- `memory/postgres_store.py` — `asyncpg` pool. Env: `POSTGRES_DSN`. Tables: `agent_commits`, `entity_links`.
- MongoDB stays as-is for messages/facts.

### 7. State additions — `state.py`

Add: `run_id`, `plan`, `agent_versions`, `blackboard`, `blackboard_snapshot`, `replan_count`, `max_replans`, `partial`.

---

## Files to modify vs create

**Create:**
- `registry/__init__.py`, `registry/agent_meta.py`, `registry/registry.py`
- `planner/__init__.py`, `planner/dag.py`, `planner/planner_agent.py`
- `orchestrator/__init__.py`, `orchestrator/orchestrator.py`, `orchestrator/blackboard.py`
- `gate/__init__.py`, `gate/completion_gate.py`
- `synthesizer/__init__.py`, `synthesizer/synthesizer.py`
- `memory/redis_store.py`, `memory/postgres_store.py`

**Modify:**
- `state.py`, `graph/graph_builder.py`, `main.py`, `agents/weather_agent.py`, `agents/outage_agent.py`, `.env.example`

**Delete (after verification):**
- `agents/router_agent.py`

---

## Reuse map

- `BaseAgent` composition + `answer_with()` — Planner, Synthesizer.
- `Observable` + `_traced_methods` — Orchestrator, Gate.
- `MCPClient` — unchanged.
- `get_mongo_store()` lifespan pattern — copied for Redis + Postgres.
- `re.search(r'\{.*?\}', raw, re.DOTALL)` JSON-from-LLM parsing — reused in Planner.
- `set_final_view(state, text, view)` — Synthesizer preserves frontend view contract.

---

## Verification

1. **Single-intent regression** — `"weather in Paris"` produces same output via new pipeline.
2. **Single-intent outage** — `"tell me about outage 17299126"` preserves `view: {type: "outage_detail"}`.
3. **Multi-intent fan-out** — `"weather in Paris and outage 17299126"` → 2 parallel tasks → merged response.
4. **Re-plan loop** — agent error → gate routes back to Planner → second attempt succeeds.
5. **Durable commit** — `output_schema.persist=true` field → row in `agent_commits`.
6. **Observability** — MLflow parent span contains Planner/Orchestrator/Agent/Gate/Synthesizer children, tagged with `run_id`.
7. **Static check** — `python -c "from graph.graph_builder import build_graph; build_graph()"` succeeds; DAG cycle test raises.
