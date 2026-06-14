# BaseAgentFramework — Architecture & Workflow

A multi-agent orchestration framework where a user request is **planned** into a
DAG of subtasks, **decomposed** into dependency waves, **executed** against
independently-running agent services discovered at runtime, and **synthesized**
into one answer — with a completion gate that can re-plan on failure.

This document explains how the whole system fits together and what happens, step
by step, when a request comes in. For setup/run commands see [README.md](README.md).

---

## 1. The big picture

The framework is **service-oriented**: the orchestration brain (the main app) and
the agents that do the work run as **separate processes** and talk over HTTP.
Agents are not hardcoded into the app — they **self-register** with a standalone
**Registry/Discovery Service** when they start, and the planner **discovers**
them at request time.

```
                                   ┌──────────────────────────────────────────┐
                                   │            MAIN APP  (:8000)              │
   user ──/chat──▶  FastAPI ──────▶│   LangGraph state machine                │
                                   │   planner → orchestrator → executor →     │
                                   │   gate → synthesizer                      │
                                   └───────┬───────────────────────┬──────────┘
                                discover   │                       │ invoke (HTTP POST /invoke)
                                  GET       │                       │
                                /agents     ▼                       ▼
                   ┌───────────────────────────────┐     ┌─────────────────────┐
                   │  REGISTRY SERVICE  (:8002)     │◀────│  Weather agent :8010 │──┐
                   │  FastAPI + MongoDB             │ reg │  Outage  agent :8011 │  │ MCP tools
                   │  TTL liveness on heartbeat     │ hb  └─────────────────────┘  │
                   └───────────────────────────────┘                              ▼
                                                                   ┌─────────────────────────┐
                                                                   │   MCP server  (:8001)    │
                                                                   │  get_weather, outage_*   │
                                                                   └─────────────────────────┘
```

| Process | Port | Responsibility |
|---|---|---|
| **Main app** | 8000 | HTTP API (`/chat`, `/chat/trace`, `/registry`) + the LangGraph pipeline |
| **Registry service** | 8002 | Agent discovery: register / heartbeat / list. MongoDB-backed, TTL liveness |
| **Agent services** | 8010, 8011, … | One process per agent; self-register, expose `POST /invoke` |
| **MCP server** | 8001 | Tool backend (weather + outage data) the agents call |

Supporting stores: **MongoDB** (conversation memory + the registry), optional
**Redis** (hot blackboard mirror), optional **Postgres** (durable commits),
**MLflow** (tracing).

---

## 2. The pipeline (LangGraph state machine)

The heart of the app is a [LangGraph](https://langchain-ai.github.io/langgraph/)
`StateGraph` built in [graph/graph_builder.py](graph/graph_builder.py). A single
typed dict — `AgentState` ([state.py](state.py)) — flows through five nodes; each
node returns a partial state update.

```
        ┌─────────┐   ┌──────────────┐   ┌──────────┐   ┌──────┐   ┌─────────────┐
START ─▶│ planner │─▶ │ orchestrator │─▶ │ executor │─▶ │ gate │─▶ │ synthesizer │─▶ END
        └─────────┘   └──────────────┘   └──────────┘   └──┬───┘   └─────────────┘
              ▲                                            │
              └──────────────── replan ────────────────────┘
```

### Node 1 — Planner ([planner/planner_agent.py](planner/planner_agent.py))
Turns the user's prompt into a DAG of subtasks.
- **Discovers agents** by calling the registry client → `GET /agents`, then renders
  a *capability menu* (each agent's id, description, tags, inputs, SLA) into the
  LLM system prompt. New agents appear automatically — no code change.
- The LLM emits JSON: `{"subtasks":[{"id","agent_id","args","depends_on","sla_ms"}]}`.
- The planner **validates** every subtask against the discovered metadata
  (`validate_args`, agent-id normalization) and drops anything unmatched.
- Output: `state["plan"]` (a `Plan` of `Subtask`s).

### Node 2 — Orchestrator ([orchestrator/orchestrator.py](orchestrator/orchestrator.py))
Decomposes the DAG into **dependency waves** using Kahn's algorithm
([planner/dag.py](planner/dag.py)). Tasks with no unresolved dependencies land in
the same wave and will run together. Detects cycles. Output: `state["waves"]`
(lists of task ids). No agents run here.

### Node 3 — Executor ([orchestrator/executor.py](orchestrator/executor.py))
Runs the waves. For each wave, all tasks run **concurrently** (`asyncio.gather`);
the next wave starts only after the current one finishes.

For each task the executor:
1. Looks up the agent's live `endpoint` via the registry client (cached).
2. `POST {endpoint}/invoke` with the task args + a context block
   `{thread_id, run_id, blackboard}`. The per-task timeout is the agent's `sla_ms`.
3. Writes the result (or error) to the **blackboard** — a shared task workspace
   ([orchestrator/blackboard.py](orchestrator/blackboard.py)). When `REDIS_URL` is
   set, each write is also mirrored to Redis (`bb:{thread_id}:{run_id}:{task_id}`,
   24h TTL) and is readable via `GET /blackboard/{thread_id}/{run_id}`; with Redis
   off this simply no-ops.

One retry per task; timeouts, HTTP errors, unknown/endpoint-less agents, and
agent-reported errors all land on the blackboard as `{"error": ...}` rather than
crashing the wave.

### Node 4 — Completion Gate ([gate/completion_gate.py](gate/))
Inspects the blackboard and decides what's next:
- **All tasks satisfied** → route to **synthesizer**.
- **Errors present and re-plan budget remains** → route back to **planner** with a
  blackboard snapshot + reason (`replan`). Successful tasks are preserved so they
  don't re-run.
- **Budget exhausted** → synthesize a `[PARTIAL]` answer from whatever succeeded.

The routing function is `route_after_gate` in
[graph/graph_builder.py](graph/graph_builder.py).

### Node 5 — Synthesizer ([synthesizer/synthesizer.py](synthesizer/synthesizer.py))
Composes one user-facing answer from the blackboard:
- Empty plan → a friendly clarification.
- Exactly one task with a structured `view` → pass it straight through (preserves
  the `{response, view}` contract for the UI).
- Otherwise → an LLM merges the successful outputs into prose, marking any errored
  section `[PARTIAL]`.
- Also **persists** any output fields whose schema marks `persist=true` to Postgres
  (best-effort; skipped if disabled).

---

## 3. Agent discovery & liveness

This is what replaced the old in-process static registry dict.

### Registration (agent → registry)
On startup, every agent service (the harness in
[baseagent/agent_server.py](baseagent/agent_server.py)) computes its advertised
`endpoint`, assigns an `instance_id`, and `POST /register`s its `AgentMeta` to the
registry. Registration is **fail-soft**: if the registry is down the agent still
starts serving and keeps retrying.

### Heartbeat & TTL liveness
Each agent heartbeats on an interval (≈ `REGISTRY_TTL_SECONDS / 3`). The registry
([registry/store.py](registry/store.py)) stores records in MongoDB with a **TTL
index** on `last_heartbeat`. A record is considered **live** iff
`status == "active" AND last_heartbeat is within the TTL window`. If an agent
crashes (no graceful deregister), its record simply ages out and disappears from
discovery. If the registry was swept/restarted, the next heartbeat comes back
`known: false` and the agent **re-registers** automatically.

> Restarting an agent registers a **new** `instance_id`; the old one lingers until
> its TTL expires (hard kill) or is removed immediately (graceful shutdown
> deregisters). The store keys by `instance_id`, so multiple live instances of the
> same `agent_id` are supported (replicas) — the executor picks one.

### Discovery (planner/executor → registry)
The planner and executor share one
[registry client](registry/registry_client.py) (`get_registry_client()` singleton)
with a short **TTL cache** (default 5s) so a single request makes ~one HTTP call
even though the planner renders the menu and validates N subtasks. If the registry
is unreachable, the client serves the last good cache (`REGISTRY_SERVE_STALE=1`) or
raises `RegistryUnavailable`, which the planner turns into a clean error instead of
a crash.

**Registry endpoints** ([registry/service.py](registry/service.py)):
`POST /register`, `POST /heartbeat`, `POST /deregister`, `GET /agents`
(+`?agent_id=`/`?tag=` filters), `GET /agents/{agent_id}`, `GET /health`.
Optional bearer auth via `REGISTRY_AUTH_TOKEN`.

---

## 4. What an agent is

An agent is a `BaseAgent` subclass ([baseagent/base_agent.py](baseagent/base_agent.py))
plus an `AgentMeta` descriptor ([registry/agent_meta.py](registry/agent_meta.py)).

```python
# agents/weather_agent.py
class WeatherAgent(BaseAgent):
    system_prompt = "You are a helpful weather reporter for a travel assistant."
    tool_names = ["get_weather"]                 # which MCP tools to load

    def run(self, state):                        # called per subtask
        city = (state.get("location") or "").lower().strip()
        return self.answer_with_tool(
            state, tool_name="get_weather", args={"city": city},
            format_text=lambda report: f"Here's the weather for {city.title()}: {report}",
        )

META = AgentMeta(
    agent_id="weather", version="1.0.0",
    capability_tags=["weather", "forecast", "city"],
    description="Reports current weather conditions for a named city.",
    input_schema={"location": FieldSpec(type="string", required=True)},
    output_schema={"text": FieldSpec(type="string")},
    sla_ms=4000,
)

if __name__ == "__main__":                       # run as its own service
    from baseagent.agent_server import run_agent
    run_agent(WeatherAgent, META)
```

`BaseAgent` composes an **LLM client** (OpenAI / OpenAI-compatible), an **MCP
client** (loads the named tools from the MCP server), and **memory** — so an agent
class only declares *what to do*, not the plumbing. `answer_with` / `answer_with_tool`
handle the increment / exception capture / tracing / state-write bookkeeping.

### Inside `/invoke`
The harness reconstructs the per-task `AgentState` with the shared
[build_task_state](orchestrator/task_state.py) helper (args spread as top-level
keys, blackboard attached), runs `agent.run(state)` on a worker thread, and maps
the result to the wire contract:

```
POST {endpoint}/invoke
  { "task_id", "agent_id", "args": {...},
    "context": { "thread_id", "run_id", "blackboard": {...} } }
→ { "text": str|null, "view": dict|null, "error": str|null }
```

The same `build_task_state` is the single source of truth, so the remote path can
never drift from the original in-process behavior.

---

## 5. End-to-end example

Request: **"Weather in Tokyo and the top outages."**

1. **Planner** discovers `weather` + `outage` from the registry, renders the menu,
   and the LLM returns two independent subtasks:
   ```json
   {"subtasks":[
     {"id":"t1","agent_id":"weather","args":{"location":"tokyo"},"depends_on":[]},
     {"id":"t2","agent_id":"outage","args":{},"depends_on":[]}]}
   ```
2. **Orchestrator** sees no dependencies → one wave: `[["t1","t2"]]`.
3. **Executor** fires both `POST /invoke`s concurrently:
   - `:8010/invoke` (weather) → calls MCP `get_weather` → `"Humid, 28°C…"`
   - `:8011/invoke` (outage) → calls MCP `list_outage_ids` → top-5 list
   - both written to the blackboard.
4. **Gate** sees both satisfied, no errors → synthesize.
5. **Synthesizer** merges → *"The current weather in Tokyo is humid… Additionally,
   there are 199 total outages, with the top 5 highlighted."*

If, say, the outage agent had timed out, the gate would route back to the planner
(if budget remained) or the synthesizer would mark that section `[PARTIAL]`.

---

## 6. HTTP API (main app)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | `{message, thread_id}` → `{response, view}` |
| `POST` | `/chat/trace` | Same pipeline but returns a step-by-step trace for the UI |
| `GET` | `/registry` | Discovered agents + liveness (powers the discovery panel) |
| `GET` | `/health` | Liveness |
| `GET` | `/state/{thread_id}` | LangGraph checkpoint snapshot |

The **trace UI** at `/trace.html` animates planner → orchestrator → executor →
gate → synthesizer and shows the **live agent-discovery panel** (each agent's
green "● LIVE" dot + `/invoke` endpoint, polled every 5s).

---

## 7. Running it

Multi-process. Easiest is the launcher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-all.ps1
```

…which starts (in order, on their own ports): MCP (8001) → Registry (8002) →
Weather (8010) + Outage (8011) → App (8000). Then open
<http://127.0.0.1:8000/trace.html>. See [README.md](README.md) for the manual
per-terminal commands, environment variables, and how to add a new agent.

> **Local tip:** if no MLflow server is running, point `MLFLOW_TRACKING_URI` at a
> local file store (`file:./mlruns`) so the app and agents don't block on tracing
> retries at startup.

---

## 8. Repository map (workflow-relevant)

```
main.py                         FastAPI app: /chat, /chat/trace, /registry; builds the graph
graph/graph_builder.py          LangGraph wiring + gate routing
state.py                        AgentState typed dict (the shared state object)
planner/
  planner_agent.py              Discover agents → LLM → validated DAG
  dag.py                        Plan/Subtask, Kahn wave computation, cycle detection
orchestrator/
  orchestrator.py               DAG → dependency waves
  executor.py                   Run waves; POST /invoke per task; write blackboard
  blackboard.py                 Shared task workspace (+ optional Redis mirror)
  task_state.py                 Shared per-task AgentState builder
gate/                           Completion gate: done? errors? re-plan budget?
synthesizer/synthesizer.py      Merge blackboard → one answer; persist durable fields
registry/
  agent_meta.py                 AgentMeta + FieldSpec (incl. endpoint/liveness fields)
  service.py                    Standalone discovery service (:8002)
  store.py                      MongoDB store with TTL liveness
  registry_client.py            httpx client + TTL cache used by planner/executor
  contracts.py                  register/heartbeat/list request+response models
baseagent/
  agent_server.py               Harness: /invoke + self-register + heartbeat
  base_agent.py                 BaseAgent core (LLM + MCP + memory composition)
agents/
  weather_agent.py              Reference agent (runs as its own service)
  outage_agent.py               Reference agent (runs as its own service)
mcp_server/weather_server.py    MCP tool backend (:8001)
frontend/trace.html             Pipeline tracer + live discovery panel
scripts/run-all.ps1             Launch the whole stack
```
