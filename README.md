# agenticworkflow

A multi-agent orchestration framework built on **FastAPI** + **LangGraph**.
The user sends one prompt; a **Planner** turns it into a directed acyclic
graph (DAG) of subtasks; an **Orchestrator** runs those subtasks in
dependency waves; a **Completion Gate** decides whether to keep going or
re-plan; a **Synthesizer** composes one final answer.

Agents run as **independent services**: each one self-registers with a
standalone **Registry/Discovery Service** on startup and is invoked over HTTP.
All results land on a shared **blackboard**, every component emits **MLflow**
spans, and a companion **/trace.html** UI animates every phase step-by-step so
you can watch the pipeline execute — including a **live agent-discovery panel**.

> For a deeper, end-to-end walkthrough of how everything fits together, see
> **[WORKFLOW.md](WORKFLOW.md)**.

---

## How a request flows

```
POST /chat
   │
   ▼
START ─▶ Planner ─▶ Orchestrator ─▶ Executor ─▶ Gate ──┬──▶ Synthesizer ──▶ END
                                                       │
                                                       └──▶ Planner   (re-plan loop)
```

1. **Planner** reads the user prompt and **discovers** the available agents by
   querying the Registry Service, then emits a JSON DAG:
   `{"subtasks": [{"id","agent_id","args","depends_on"}]}`. The LLM picks which
   discovered agents to call, in what order, and with what args, and the planner
   validates each subtask against the agent's schema. (`planner/planner_agent.py`)
2. **Orchestrator** computes dependency waves with Kahn's algorithm. Independent
   tasks share a wave; the decomposition is handed to the Executor. No agents
   run here. (`orchestrator/orchestrator.py`)
3. **Executor** runs each wave concurrently via `asyncio.gather`, invoking each
   chosen agent over HTTP at `POST {endpoint}/invoke` (endpoint looked up from
   the registry, timeout = the agent's `sla_ms`). Every task result — success or
   error — is written to a shared blackboard. (`orchestrator/executor.py`)
4. **Completion Gate** inspects the blackboard: all subtasks satisfied?
   any errors? re-plan budget left (default 3)? Routes back to Planner
   on partial failure, or forward to Synthesizer otherwise. (`gate/`)
5. **Synthesizer** reads the whole blackboard, composes one user-facing
   answer (marking sections `[PARTIAL]` where agents errored), and
   optionally commits durable fields to Postgres. (`synthesizer/`)

The outer flow is a small LangGraph; the DAG itself executes inside the
Executor node, so there is no per-request dynamic graph mutation.

---

## Why this shape

| Concern | Solution in this repo |
| --- | --- |
| Add a new capability without touching the planner | Write a `BaseAgent` subclass with an `AgentMeta` block and run it as its own service; it self-registers with the registry. Planner discovers it next request. |
| Scale / deploy agents independently | Each agent is its own process with its own port and `/invoke` endpoint; the Executor calls it over HTTP. Run multiple instances of one `agent_id` and the registry tracks them as live replicas. |
| Detect a dead agent | Agents heartbeat to the registry; records expire via a MongoDB TTL when an agent stops heartbeating, so the planner stops offering it. |
| Run independent subtasks in parallel | Planner emits `depends_on=[]` on independent tasks; Orchestrator computes waves and the Executor runs each wave via `asyncio.gather`. |
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

if __name__ == "__main__":
    from baseagent.agent_server import run_agent
    run_agent(WeatherAgent, META)
```

Each agent runs as its **own service**. On startup it self-registers its
`AgentMeta` (plus its `/invoke` endpoint) with the **Registry Service**
(`registry/service.py`, :8002) and heartbeats to stay "live" (records expire on
TTL if an agent crashes). The Planner **discovers** agents by querying the
registry and renders the live menu into its system prompt; the Executor invokes
each chosen agent over HTTP at `POST {endpoint}/invoke`. There is no longer a
static in-process registry dict.

---

## Memory backends

| Store | Purpose | Required? |
| --- | --- | --- |
| **MongoDB** | Short-term message history (24h TTL) + long-term per-thread facts; also backs the agent registry (TTL liveness) | Yes |
| **Redis** | Hot blackboard mirror keyed by `bb:{thread_id}:{run_id}:{task_id}` (24h TTL); read back via `GET /blackboard/{thread_id}/{run_id}` | No — no-ops when `REDIS_URL` unset |
| **Postgres** | Durable commits for `output_schema` fields marked `persist=true`; audit + entity_links tables | No — no-ops when `POSTGRES_DSN` unset |

Redis and Postgres are intentionally optional: the framework runs end-to-end
with just MongoDB. Tables are created automatically on Postgres startup.

**Enable the Redis blackboard mirror:** `pip install -r requirements.txt` (includes
`redis`), start a Redis server, then set `REDIS_URL=redis://localhost:6379/0` in `.env`.
Every blackboard write is then mirrored to Redis and readable via
`GET /blackboard/{thread_id}/{run_id}`. With `REDIS_URL` unset the mirror simply no-ops.

---

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/chat` | Send a prompt; returns `{response, view}` |
| POST | `/chat/trace` | Same as `/chat` but returns every intermediate node update for the visualization UI |
| GET | `/registry` | Live discovery: every agent currently registered + its schema, endpoint, and last heartbeat (proxies the Registry Service) |
| GET | `/state/{thread_id}` | LangGraph checkpoint snapshot for a thread |
| GET | `/blackboard/{thread_id}/{run_id}` | Read back the Redis-mirrored blackboard entries for a run (empty when Redis is disabled) |
| GET | `/health` | Liveness probe |

`POST /chat` body:

```json
{ "message": "Show me the top 5 outages", "thread_id": "user-123" }
```

---

## Execution Tracer UI

`/trace.html` is a standalone single-page visualizer that animates every
request:

- **Discovery panel** — the agents the planner discovered from the registry,
  each with a green **● LIVE** dot and its `/invoke` endpoint, polled every 5s so
  agents appearing/expiring show up without a reload.
- **Planner card** — shows the agent menu the LLM saw and the resulting plan
  (every subtask with id, agent, args, deps).
- **Orchestrator / Executor cards** — wave decomposition (Wave 1, Wave 2, …) plus
  the blackboard table after the Executor invokes each agent. Green rows =
  success, red = error.
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
│  └─ graph_builder.py           # LangGraph: planner → orchestrator → executor → gate → synthesizer
├─ planner/
│  ├─ dag.py                     # Plan, Subtask, wave computation (Kahn's), cycle detection
│  └─ planner_agent.py           # LLM-based DAG emitter; discovers agents via the registry
├─ orchestrator/
│  ├─ orchestrator.py            # DAG → dependency waves (Kahn's)
│  ├─ executor.py                # Runs waves; POST /invoke per task; one retry per task
│  ├─ task_state.py              # Shared per-task AgentState builder
│  └─ blackboard.py              # In-memory + Redis-mirrored shared workspace
├─ gate/
│  └─ completion_gate.py         # Done? Errors? Re-plan budget left?
├─ synthesizer/
│  └─ synthesizer.py             # Merges blackboard → one answer; commits persistable fields
├─ registry/
│  ├─ agent_meta.py              # AgentMeta + FieldSpec pydantic models (+ endpoint/liveness)
│  ├─ contracts.py               # Register/heartbeat/list request+response models
│  ├─ store.py                   # MongoDB-backed registry store (TTL liveness)
│  ├─ service.py                 # Standalone discovery service (:8002)
│  └─ registry_client.py         # httpx client + TTL cache used by Planner/Executor
├─ agents/                       # Each runs as its own self-registering service
│  ├─ weather_agent.py           # Reference agent: weather report by city
│  └─ outage_agent.py            # Reference agent: grid-outage list / detail view
├─ baseagent/                    # Reusable agent core (composition, not inheritance)
│  ├─ agent_server.py            # Harness: /invoke + self-register + heartbeat
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
| `REGISTRY_URL` | Registry/discovery service base URL (default `http://127.0.0.1:8002`). |
| `REGISTRY_AUTH_TOKEN` | Optional bearer token for the registry service. |
| `REGISTRY_TTL_SECONDS` | Liveness window; a record expires without a heartbeat (default 90). |
| `AGENT_PORT` | Port an agent service binds to (each agent on its own port). |
| `MONGODB_URI` | MongoDB connection string (also backs the registry). |
| `MLFLOW_TRACKING_URI` | MLflow server URL. Use a `file://` URI to keep it local. |
| `REDIS_URL` | Optional; enables hot blackboard mirror. |
| `POSTGRES_DSN` | Optional; enables durable commit store. |

### Run

The system is multi-process. Each piece runs on its own port:

| Service | Port | Start command |
| --- | --- | --- |
| MCP tool server | 8001 | `python -m mcp_server.weather_server` |
| Registry / discovery service | 8002 | `python -m registry.service` |
| Weather agent service | 8010 | `$env:AGENT_PORT="8010"; python -m agents.weather_agent` |
| Outage agent service | 8011 | `$env:AGENT_PORT="8011"; python -m agents.outage_agent` |
| FastAPI app | 8000 | `python main.py` |

The easiest way is the launcher, which opens each in its own window in the right
order (registry up before agents register; agents up before the app queries them):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-all.ps1
```

Or start them by hand:

```powershell
# Terminal 1 — MCP server on :8001
python -m mcp_server.weather_server

# Terminal 2 — Registry/discovery service on :8002
python -m registry.service

# Terminal 3 — Weather agent service on :8010
$env:AGENT_PORT="8010"; python -m agents.weather_agent

# Terminal 4 — Outage agent service on :8011
$env:AGENT_PORT="8011"; python -m agents.outage_agent

# Terminal 5 — FastAPI app on :8000
python main.py
```

Then open:

- <http://127.0.0.1:8000> — chat UI
- <http://127.0.0.1:8000/trace.html> — execution tracer (recommended starting point)

> **Tip:** if you don't have an MLflow tracking server running, set
> `MLFLOW_TRACKING_URI=file:./mlruns` (a local file store). Otherwise every
> process — the app *and* each agent — blocks on connection retries at startup
> and during the first request.

---

## Extending

### Add a new agent

1. Create `agents/my_agent.py` inheriting from `BaseAgent`.
2. Set `system_prompt` and (optionally) `tool_names`.
3. Implement `run(state)` — or call `self.answer_with_tool(...)` /
   `self.answer_with(...)` for the one-shot template.
4. Declare a module-level `META = AgentMeta(...)` and add the
   `if __name__ == "__main__": run_agent(MyAgent, META)` block at the bottom.
5. Run it as its own service on a free port:
   `$env:AGENT_PORT="8012"; python -m agents.my_agent` (and add a line to
   `scripts/run-all.ps1`).

On startup the agent self-registers with the registry service, so the Planner
discovers it on the next request — no planner or graph code change required.

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
