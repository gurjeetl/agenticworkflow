# agenticworkflow

A multi-agent orchestration framework built on **FastAPI** + **LangGraph**.
The user sends one prompt; a **Planner** turns it into a directed acyclic
graph (DAG) of subtasks; an **Orchestrator** runs those subtasks in
dependency waves; a **Completion Gate** decides whether to keep going or
re-plan; a **Synthesizer** composes one final answer.

Agents are registered through a typed **agent registry**, all results land
on a shared **blackboard**, every component emits **MLflow** spans, and a
companion **/trace.html** UI animates every phase step-by-step so you can
watch the pipeline execute.

---

## How a request flows

```
POST /chat
   │
   ▼
START ──▶ Planner ──▶ Orchestrator ──▶ Gate ──┬──▶ Synthesizer ──▶ END
                                              │
                                              └──▶ Planner   (re-plan loop)
```

1. **Planner** reads the user prompt and the full agent registry, then
   emits a JSON DAG: `{"subtasks": [{"id","agent_id","args","depends_on"}]}`.
   The LLM picks which registered agents to call, in what order, and with
   what args. (`planner/planner_agent.py`)
2. **Orchestrator** computes dependency waves with Kahn's algorithm and
   fans each wave out via `asyncio.gather`. Every task result — success
   or error — is written to a shared blackboard. (`orchestrator/`)
3. **Completion Gate** inspects the blackboard: all subtasks satisfied?
   any errors? re-plan budget left (default 3)? Routes back to Planner
   on partial failure, or forward to Synthesizer otherwise. (`gate/`)
4. **Synthesizer** reads the whole blackboard, composes one user-facing
   answer (marking sections `[PARTIAL]` where agents errored), and
   optionally commits durable fields to Postgres. (`synthesizer/`)

The outer flow is a small LangGraph; the DAG itself executes inside the
Orchestrator node, so there is no per-request dynamic graph mutation.

---

## Why this shape

| Concern | Solution in this repo |
| --- | --- |
| Add a new capability without touching the planner | Drop a `BaseAgent` subclass with an `AgentMeta` block in `agents/`; it auto-registers on import. Planner sees it next request. |
| Run independent subtasks in parallel | Planner emits `depends_on=[]` on independent tasks; Orchestrator computes waves and runs each wave via `asyncio.gather`. |
| Recover from a transient agent failure | Gate sees the error entry on the blackboard, routes back to Planner with a snapshot + reason. Capped by `max_replans`. |
| Audit who said what | Every node/agent run is wrapped in an MLflow span tagged with `run_id`, `agent_id`, `version`. |
| See execution end-to-end | Open `/trace.html`, type a prompt, step through Planner → Orchestrator → Gate → Synthesizer with the actual blackboard state visible at each phase. |

---

## Agent registry

Every agent declares an `AgentMeta` block:

```python
# agents/weather_agent.py
META = AgentMeta(
    agent_id="weather",
    version="1.0.0",
    capability_tags=["weather", "forecast", "city"],
    description="Reports current weather conditions for a named city.",
    input_schema={"location": FieldSpec(type="string", required=True)},
    output_schema={"text": FieldSpec(type="string")},
    sla_ms=4000,
)
register(META, WeatherAgent)
```

The Planner renders this menu into its system prompt so the LLM can match
the user's intent to the right agent. To add a new agent: write the class,
write the `META`, call `register()`, import the module in
`graph/graph_builder.py`. That's it.

---

## Memory backends

| Store | Purpose | Required? |
| --- | --- | --- |
| **MongoDB** | Short-term message history (24h TTL) + long-term per-thread facts | Yes |
| **Redis** | Hot blackboard mirror keyed by `bb:{thread_id}:{run_id}:{task_id}` with TTL | No — no-ops when `REDIS_URL` unset |
| **Postgres** | Durable commits for `output_schema` fields marked `persist=true`; audit + entity_links tables | No — no-ops when `POSTGRES_DSN` unset |

Redis and Postgres are intentionally optional: the framework runs end-to-end
with just MongoDB. Tables are created automatically on Postgres startup.

---

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/chat` | Send a prompt; returns `{response, view}` |
| POST | `/chat/trace` | Same as `/chat` but returns every intermediate node update for the visualization UI |
| GET | `/registry` | Dump every registered agent with its schema |
| GET | `/state/{thread_id}` | LangGraph checkpoint snapshot for a thread |
| GET | `/health` | Liveness probe |

`POST /chat` body:

```json
{ "message": "Show me the top 5 outages", "thread_id": "user-123" }
```

---

## Execution Tracer UI

`/trace.html` is a standalone single-page visualizer that animates every
request:

- **Planner card** — shows the agent registry the LLM saw (the menu) and
  the resulting plan (every subtask with id, agent, args, deps).
- **Orchestrator card** — wave decomposition (Wave 1, Wave 2, …) plus the
  blackboard table after execution. Green rows = success, red = error.
- **Gate card** — the SYNTHESIZE / REPLAN decision with reasoning.
- **Synthesizer card** — final user-facing answer, with `[PARTIAL]` badge
  if anything errored.

Controls: **Step ▶** advance one phase, **Reveal all**, **Auto-play**,
**Restart**. Four sample prompts cover single-intent, detail-view, and
multi-intent fan-out scenarios.

---

## Project layout

```
agenticworkflow/
├─ main.py                       # FastAPI app, /chat, /chat/trace, /registry
├─ state.py                      # AgentState TypedDict (shared blackboard)
├─ graph/
│  └─ graph_builder.py           # LangGraph: planner → orchestrator → gate → synthesizer
├─ planner/
│  ├─ dag.py                     # Plan, Subtask, wave computation (Kahn's), cycle detection
│  └─ planner_agent.py           # LLM-based DAG emitter with registry-aware prompt
├─ orchestrator/
│  ├─ orchestrator.py            # Runs DAG waves via asyncio.gather, one retry per task
│  └─ blackboard.py              # In-memory + Redis-mirrored shared workspace
├─ gate/
│  └─ completion_gate.py         # Done? Errors? Re-plan budget left?
├─ synthesizer/
│  └─ synthesizer.py             # Merges blackboard → one answer; commits persistable fields
├─ registry/
│  ├─ agent_meta.py              # AgentMeta + FieldSpec pydantic models
│  └─ registry.py                # Process-wide AGENT_REGISTRY + register/get/match helpers
├─ agents/
│  ├─ weather_agent.py           # Reference agent: weather report by city
│  └─ outage_agent.py            # Reference agent: grid-outage list / detail view
├─ baseagent/                    # Reusable agent core (composition, not inheritance)
│  ├─ base_agent.py              # BaseAgent + answer_with helpers
│  ├─ llm_client.py              # ChatOpenAI wrapper + tool execution
│  ├─ mcp_client.py              # MCP transport + tool loading
│  ├─ agent_memory.py            # Sliding-window message trim + fact persistence
│  └─ events.py                  # Event name constants
├─ memory/
│  ├─ mongo_store.py             # MongoDB short-term + long-term store
│  ├─ redis_store.py             # Optional hot blackboard mirror
│  └─ postgres_store.py          # Optional durable commit + entity_links store
├─ observability/
│  ├─ observable.py              # Observable base + MLflow span auto-wrapping
│  └─ mlflow_setup.py            # MLflow init
├─ mcp_server/
│  └─ weather_server.py          # Local MCP server: get_weather + outage tools
├─ frontend/
│  ├─ index.html                 # Chat UI
│  └─ trace.html                 # Step-by-step execution tracer UI
└─ docs/PLAN_PLANNER_ORCHESTRATOR.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- A running MongoDB (defaults to `mongodb://localhost:27017`)
- An OpenAI API key (or any OpenAI-compatible endpoint via `OPENAI_BASE_URL`)
- Optional: MLflow, Redis, Postgres

### Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configure

```powershell
Copy-Item .env.example .env
# Edit .env: set OPENAI_API_KEY at minimum
```

Key environment variables:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required. API key for the LLM. |
| `OPENAI_MODEL` | Defaults to `gpt-4o-mini`. |
| `OPENAI_BASE_URL` | Optional override for OpenAI-compatible providers. |
| `MCP_SERVER_URL` | MCP server URL. Leave empty to skip MCP tool loading. |
| `MCP_TRANSPORT` | `sse` \| `stdio` \| `websocket` \| `streamable_http`. |
| `MCP_AUTH_TOKEN` | Optional bearer token for the MCP server. |
| `MONGODB_URI` | MongoDB connection string. |
| `MLFLOW_TRACKING_URI` | MLflow server URL. Use a `file://` URI to keep it local. |
| `REDIS_URL` | Optional; enables hot blackboard mirror. |
| `POSTGRES_DSN` | Optional; enables durable commit store. |

### Run

Two processes — the MCP tool server and the FastAPI app:

```powershell
# Terminal 1 — MCP server on :8001
python -m mcp_server.weather_server

# Terminal 2 — FastAPI app on :8000
python main.py
```

Then open:

- <http://127.0.0.1:8000> — chat UI
- <http://127.0.0.1:8000/trace.html> — execution tracer (recommended starting point)

---

## Extending

### Add a new agent

1. Create `agents/my_agent.py` inheriting from `BaseAgent`.
2. Set `system_prompt` and (optionally) `tool_names`.
3. Implement `run(state)` — or call `self.answer_with_tool(...)` /
   `self.answer_with(...)` for the one-shot template.
4. Declare a module-level `META = AgentMeta(...)` and call
   `register(META, MyAgent)` at the bottom of the file.
5. Add `import agents.my_agent` to `graph/graph_builder.py` so the
   module loads (and self-registers) at startup.

The Planner will pick up the new agent on the next request — no planner
code change required.

### Add a new MCP tool

Add an `@mcp.tool()`-decorated function to
`mcp_server/weather_server.py` (or stand up a separate MCP server and
point `MCP_SERVER_URL` at it). Any agent that names the tool in its
`tool_names` list will get it bound automatically.

### Use a different LLM provider

Set `OPENAI_BASE_URL` and (if needed) `OPENAI_MODEL` — any
OpenAI-compatible endpoint works without code changes.

---

## Observability

Every Planner, Orchestrator, Gate, Synthesizer, and Agent invocation is
auto-wrapped in an MLflow span by the `Observable` base class
(`observability/observable.py`). Spans capture:

- `run_id` — the per-request UUID propagated through every step
- `agent_id`, `agent_version`, `wave`, `retry_count` (on agent spans)
- structured events for tool calls, blackboard writes, gate decisions

Point `MLFLOW_TRACKING_URI` at a tracking server to persist traces, or
use a `file:///path/to/mlruns_local` URI for local-only tracing.
