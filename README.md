# Genie Platform

A multi-agent **platform** built on **FastAPI** + **LangGraph**. It provides a
router → synthesizer workflow plus reusable capabilities — MCP tool connectivity,
agent registry/discovery, A2A messaging, multi-layer memory, content-safety
guards, and observability — and lets **applications** build agents by inheriting
a single `BaseAgent`.

The user sends one prompt; an **Input Guard** screens it; a **Router** triages
intent; a **Planner** turns it into a directed acyclic graph (DAG) of subtasks; an
**Orchestrator** runs those in dependency waves; a **Completion Gate** decides
whether to keep going or re-plan; a **Synthesizer** composes one answer; an
**Output Guard** screens the reply. Agents run as **independent A2A services**
that self-register with a standalone **Registry**, and every component emits
**MLflow** spans that the companion **/trace.html** UI animates step-by-step.

> For an end-to-end walkthrough see **[docs/WORKFLOW.md](docs/WORKFLOW.md)**, and
> **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the design rationale.

---

## Platform vs. application

The codebase is split along one boundary, enforced by `import-linter`
(the platform kernel may **never** import an application):

- **`src/genie/`** — the **platform kernel**: the workflow (router → … →
  synthesizer), plus the capabilities every agent gets for free (LLM client, MCP
  connectivity, registry, A2A, memory stores, guards, observability, config).
- **`src/applications/`** — **domain agents** built *on* the platform. Each
  inherits `genie.agents.base.BaseAgent`, declares an `AgentMeta`, and runs as its
  own A2A service. The bundled `demo` application ships three: `weather`,
  `outage`, and `rag`.

Adding a capability to an application never requires touching the kernel.

---

## How a request flows

```
POST /chat
   │
   ▼
START ─▶ Input Guard ─▶ Router ─┬─(fast)────────────────────────▶ Executor ─┐
                                ├─(chitchat)─────────────────────▶ Synthesizer
                                └─(plan)▶ Planner ▶ Orchestrator ▶ Executor ▶ Gate ─┬─▶ Synthesizer ─▶ Output Guard ─▶ END
                                                                                   └─▶ Planner  (re-plan loop)
```

1. **Input Guard** screens the prompt (prompt-injection, toxicity, banned topics)
   and redacts PII/secrets. Blocked prompts short-circuit to a safe refusal.
   (`genie/security/guards.py`)
2. **Router** does cheap intent triage — regex + a local embedding classifier,
   then an LLM fallback — and routes to `fast` (one obvious agent → straight to
   the Executor), `chitchat` (→ Synthesizer), or `plan` (→ full pipeline).
   (`genie/application/nodes/router.py`)
3. **Planner** discovers the live agents from the Registry and emits a JSON DAG
   `{"subtasks":[{"id","agent_id","args","depends_on"}]}`, validating each subtask
   against the agent's schema. (`genie/application/nodes/planner.py`)
4. **Orchestrator** computes dependency waves with Kahn's algorithm; independent
   tasks share a wave. (`genie/application/nodes/orchestrator.py`)
5. **Executor** runs each wave concurrently, invoking each chosen agent over
   **A2A JSON-RPC** (`message/send` to `POST {endpoint}/a2a`). Every result —
   success or error — lands on a shared **blackboard**.
   (`genie/application/nodes/executor.py`)
6. **Completion Gate** inspects the blackboard: all subtasks satisfied? errors?
   re-plan budget left (default 3)? Routes back to the Planner or onward.
   (`genie/application/nodes/completion_gate.py`)
7. **Synthesizer** merges the blackboard into one answer (marking `[PARTIAL]`
   where agents errored) and writes durable facts/commits.
   (`genie/application/nodes/synthesizer.py`)
8. **Output Guard** screens the final answer before it reaches the user.

---

## Agents inherit `BaseAgent`

An application agent is a `BaseAgent` subclass plus an `AgentMeta`. The platform
supplies the LLM client, MCP tool loading, and working memory; the agent declares
which tools it wants (`tool_names`) and what to do (`run`).

```python
# src/applications/demo/weather/agent.py
from genie.agents.base import BaseAgent
from genie.registry import AgentMeta, FieldSpec, Skill

class WeatherAgent(BaseAgent):
    system_prompt = "You are a helpful weather reporter."
    tool_names = ["get_weather"]          # MCP tools the platform binds for this agent

    def run(self, state):
        city = (state.get("location") or "").lower().strip()
        return self.answer_with_tool(
            state, tool_name="get_weather", args={"city": city},
            format_text=lambda r: f"Here's the current weather for {city.title()}: {r}",
        )

META = AgentMeta(
    agent_id="weather",
    capability_tags=["weather", "forecast", "city"],   # what the Router/Planner match on
    description="Reports current weather conditions for a named city.",
    skills=[Skill(                                      # A2A AgentSkill (served in the Agent Card)
        id="get_current_weather", name="Current weather report",
        description="Reports current weather conditions for a named city.",
        tags=["weather", "forecast", "city"],
        examples=["What's the weather in Paris?", "Weather in Tokyo"],
    )],
    input_schema={"location": FieldSpec(type="string", required=True, description="City name.")},
    output_schema={"text": FieldSpec(type="string", persist=True)},
    sla_ms=4000,
)

if __name__ == "__main__":                              # run as its own A2A service
    from genie.agents.server import run_agent
    run_agent(WeatherAgent, META)
```

`tool_names`: `None` = load all permitted MCP tools, `[]` = skip MCP entirely
(pure-LLM agents), `[...]` = load only the named tools.

### A2A discovery

Each agent runs as its own service, self-registers its `AgentMeta` with the
**Registry** (`services/registry/server.py`, :8002), and heartbeats to stay live
(records expire on TTL if it crashes). It also serves a spec-compliant **A2A
Agent Card** at `GET /.well-known/agent.json`, whose `skills` are projected from
the same `AgentMeta` — so the registry record and the card never drift.
`capability_tags` drive internal routing; `skills` are the A2A-standard
advertisement for interop.

---

## Capabilities the platform provides

| Capability | Where | Notes |
| --- | --- | --- |
| **MCP connectivity** | `genie/mcp/` | `BaseAgent` loads MCP tools from config; a single server via `MCP_SERVER_URL` or named servers via the YAML `mcp_services` block. |
| **Registry / discovery** | `genie/registry/`, `services/registry/` | Self-registration + heartbeat + TTL liveness, backed by MongoDB. |
| **A2A messaging** | `genie/a2a/` | JSON-RPC `message/send`, Agent Cards. |
| **Memory** | `genie/memory/` | See below. |
| **Content guards** | `genie/security/` | Mandatory input/output guard (llm-guard). |
| **Observability** | `genie/observability/` | MLflow spans auto-wrapped by `Observable`. |
| **Config** | `genie/platform/config.py` | Central `Settings` (pydantic-settings + YAML). |

### Memory backends

| Store | Purpose | Required? |
| --- | --- | --- |
| **MongoDB** | Session messages + durable conversations + per-thread facts + durable commits; also backs the registry (TTL liveness) | **Yes** |
| **Redis** | Hot blackboard mirror keyed by `bb:{thread_id}:{run_id}:{task_id}`; read via `GET /blackboard/{thread_id}/{run_id}` | No — no-ops when `REDIS_URL` unset |
| **Milvus** | Semantic long-term memory (embeddings the Planner recalls from) | No — no-ops when `MILVUS_URI`/`MILVUS_DB_PATH` unset |

Redis and Milvus are optional; the framework runs end-to-end with just MongoDB.

---

## HTTP API (gateway, :8000)

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/chat` | Send a prompt; returns `{response, view}` |
| POST | `/chat/trace` | Same pipeline, but returns every node update for the tracer UI |
| GET | `/registry` | Live discovery: every registered agent + schema, endpoint, last heartbeat |
| GET | `/state/{thread_id}` | LangGraph checkpoint snapshot for a thread |
| GET | `/blackboard/{thread_id}/{run_id}` | Redis-mirrored blackboard entries for a run |
| GET | `/conversations` · `/conversations/{thread_id}` | List / resume / delete past conversations |
| GET | `/health` | Liveness probe |

`POST /chat` body:

```json
{ "message": "Show me the top 5 outages", "thread_id": "user-123" }
```

---

## Project layout

```
genie-platform/
├─ src/
│  ├─ app.py                         # gateway entry: app = create_app()  (uvicorn app:app)
│  ├─ genie/                         # ── PLATFORM KERNEL ──
│  │  ├─ agents/                     # base.py (BaseAgent), server.py (A2A harness), memory, task_state
│  │  ├─ application/                # graph.py, state.py, checkpointer.py, blackboard.py
│  │  │  └─ nodes/                   # router, planner, orchestrator, executor, completion_gate, synthesizer
│  │  ├─ interface/                  # bootstrap.py (create_app) + routers/{chat,health,state,registry,conversations}
│  │  ├─ llm/                        # ChatOpenAI client wrapper
│  │  ├─ mcp/                        # MCP client, config, tool-permission filter
│  │  ├─ registry/                   # AgentMeta, contracts, MongoDB store, client
│  │  ├─ a2a/                        # JSON-RPC client, Agent Card, types
│  │  ├─ memory/                     # mongo / facts / commit / redis / vector stores
│  │  ├─ security/                   # guards.py (graph nodes) + llm_guard.py
│  │  ├─ observability/              # Observable, logging, MLflow setup
│  │  └─ platform/                   # config.py (Settings), events.py
│  └─ applications/
│     └─ demo/                       # ── DEMO APPLICATION (agents) ──
│        ├─ providers.py             # launch manifest (AgentClass, META, port)
│        ├─ weather/agent.py
│        ├─ outage/agent.py
│        └─ rag/agent.py
├─ services/
│  ├─ registry/server.py             # standalone Registry/Discovery service (:8002)
│  └─ mcp/{weather_server,rag_index}.py   # standalone MCP tool server (:8001)
├─ config/{default,test}.yaml        # pydantic-settings YAML sources (incl. mcp_services)
├─ frontend/{index,trace}.html       # chat UI + step-by-step execution tracer
├─ tests/                            # unit / integration / e2e
├─ docs/                             # ARCHITECTURE, WORKFLOW, ADRs, diagrams
├─ scripts/run-all.ps1               # launches the whole multi-process stack
└─ pyproject.toml                    # packaging (packages = src/genie, src/applications) + import-linter
```

---

## Setup

This project uses **[uv](https://docs.astral.sh/uv/)** — a fast Python package &
project manager — for dependency management, the virtual environment, and running
commands. `uv` reads `pyproject.toml` (and the resolved `uv.lock`), creates an
isolated `.venv`, and pins the right Python automatically.

### Prerequisites

- **uv** — install it first (it also manages the Python toolchain for you):
  ```powershell
  # Windows (PowerShell)
  powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
  # macOS / Linux
  # curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  (Alternatives: `pipx install uv`, `winget install astral-sh.uv`, `brew install uv`.)
- **Python 3.13** — optional to install yourself; `uv` will fetch the interpreter
  pinned in `.python-version` on first sync if one isn't found (`uv python install 3.13`).
- A running **MongoDB** (defaults to `mongodb://localhost:27017`)
- An **OpenAI API key** (or any OpenAI-compatible endpoint via `OPENAI_BASE_URL`)
- Optional: MLflow, Redis, Milvus

### Install

From the repo root:

```powershell
uv sync                     # create .venv + install genie, applications, and the
                            # editable genie-rag-contracts package from src/
```

`uv sync` resolves and installs everything declared in `pyproject.toml`
(including the local `packages/genie-rag-contracts` editable source wired up under
`[tool.uv.sources]`), creating a project `.venv` in one step — no manual
`venv`/activation needed. To include the test/lint toolchain:

```powershell
uv sync --extra dev         # adds pytest, pytest-asyncio, ruff, import-linter
```

Run any command inside the managed environment by prefixing it with `uv run`
(no activation required) — e.g. `uv run python -m ...`, `uv run uvicorn ...`,
`uv run pytest`. `uv` ensures the env is up to date before each run. If you prefer
a classic activated shell, `.venv\Scripts\Activate.ps1` (PowerShell) or
`source .venv/bin/activate` (bash) still works.

This puts `genie` and `applications` on the path. The standalone `services.*` and
the `app` entry resolve from the repo root (or set `PYTHONPATH=src`).

### Configure

```powershell
Copy-Item .env.example .env
# Edit .env: set OPENAI_API_KEY at minimum
```

Configuration is centralized in `genie.platform.config.Settings`
(pydantic-settings). It reads, in priority order: process env vars → a YAML file
(`config/default.yaml` or `$GENIE_CONFIG_FILE`) → built-in defaults. Flat values
come from `.env`; **nested** structures that env vars can't express — named MCP
servers under `mcp_services`, LLM backends under `llm_services` — live in the YAML.

### Pointing at a self-hosted LLM (`llm_services`)

To run against an OpenAI-compatible self-hosted model instead of the flat
`openai_*` config, declare it under `llm_services` in `config/default.yaml` (or
`config/local.yaml`). `default` selects which model is used for **all** LLM calls
(router, planner, synthesizer, agents); the `base_url` is derived as
`http://{host}:{port}/{prompting_path}`:

```yaml
llm_services:
  default: gpt_oss            # the model below is used for every LLM call
  models:
    gpt_oss:
      host: "genieapps4.dev.oati.local"
      port: 8033
      model_name: "openai/gpt-oss-120b"
      prompting_path: "v1"     # → base_url http://genieapps4.dev.oati.local:8033/v1
      max_token_limit: 131072  # model context window (metadata; not max output tokens)
      # api_key: "EMPTY"       # open endpoints need no key — a placeholder is applied
```

When a named backend is active, `openai_model` becomes an inert fallback (used only
if `llm_services.default` is unset). Embeddings still use the flat `openai_*` config,
so set `OPENAI_API_KEY` if semantic long-term memory (Milvus) is enabled.

Key environment variables:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required. API key for the LLM. |
| `OPENAI_MODEL` | Defaults to `gpt-4o-mini`. |
| `OPENAI_BASE_URL` | Optional override for OpenAI-compatible providers. |
| `OPENAI_TEMPERATURE` | Set to `0` for deterministic routing/planning. |
| `MCP_SERVER_URL` | MCP server URL. Leave empty to skip MCP tool loading. |
| `MCP_TRANSPORT` | `sse` \| `stdio` \| `websocket` \| `streamable_http`. |
| `MCP_AUTH_TOKEN` | Optional bearer token for the MCP server. |
| `REGISTRY_URL` | Registry/discovery base URL (default `http://127.0.0.1:8002`). |
| `REGISTRY_TTL_SECONDS` | Liveness window; a record expires without a heartbeat (default 90). |
| `AGENT_PORT` | Port an agent service binds to. |
| `MONGODB_URI` | MongoDB connection string (also backs the registry). |
| `MLFLOW_TRACKING_URI` | MLflow server URL, or a local store (see tip below). |
| `REDIS_URL` | Optional; enables the hot blackboard mirror. |
| `MILVUS_DB_PATH` / `MILVUS_URI` | Optional; enables semantic long-term memory. |

### Run

The system is multi-process; each piece runs on its own port:

| Service | Port | Start command |
| --- | --- | --- |
| MCP tool server | 8001 | `uv run python -m services.mcp.weather_server` |
| Registry / discovery | 8002 | `uv run python -m services.registry.server` |
| Weather agent | 8010 | `$env:AGENT_PORT="8010"; uv run python -m applications.demo.weather.agent` |
| Outage agent | 8011 | `$env:AGENT_PORT="8011"; uv run python -m applications.demo.outage.agent` |
| RAG agent | 8012 | `$env:AGENT_PORT="8012"; uv run python -m applications.demo.rag.agent` |
| Gateway (FastAPI) | 8000 | `uv run uvicorn app:app --host 0.0.0.0 --port 8000` |

The easiest way is the launcher, which opens each in its own window in the right
order (registry before agents register; agents before the gateway queries them):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-all.ps1
```

Then open:

- <http://127.0.0.1:8000> — chat UI
- <http://127.0.0.1:8000/trace.html> — execution tracer (recommended starting point)

> **Tip:** if you don't have an MLflow tracking server running, set
> `MLFLOW_TRACKING_URI=sqlite:///mlflow_local.db` (a local store). Otherwise every
> process blocks on connection retries at startup and during the first request.

---

## Extending

### Add a new agent

1. Create `src/applications/<app>/<name>/agent.py` inheriting `BaseAgent`.
2. Set `system_prompt` and (optionally) `tool_names`.
3. Implement `run(state)` — or use `self.answer_with_tool(...)` /
   `self.answer_with(...)` for the one-shot template.
4. Declare a module-level `META = AgentMeta(...)` with explicit A2A `skills`, plus
   the `if __name__ == "__main__": run_agent(MyAgent, META)` block.
5. Run it on a free port and add it to `scripts/run-all.ps1` (and the app's
   `providers.py` launch manifest).

On startup the agent self-registers, so the Planner discovers it next request —
no kernel change required.

### Add a new MCP tool

Add an `@mcp.tool()`-decorated function to `services/mcp/weather_server.py` (or
stand up a separate MCP server and point `MCP_SERVER_URL` / a `mcp_services` entry
at it). Any agent that names the tool in `tool_names` gets it bound automatically.

### Use a different LLM provider

Set `OPENAI_BASE_URL` and `OPENAI_MODEL` — any OpenAI-compatible endpoint works
without code changes.

---

## Observability

Every Router, Planner, Orchestrator, Gate, Synthesizer, and Agent invocation is
auto-wrapped in an MLflow span by the `Observable` base class
(`genie/observability/observable.py`), capturing the per-request `run_id`,
`agent_id`/`version`, wave/retry counts, and structured events for tool calls,
blackboard writes, and gate decisions. Point `MLFLOW_TRACKING_URI` at a tracking
server to persist traces, or use a local `sqlite:///mlflow_local.db` store.

---

## Tests & boundaries

```powershell
uv sync --extra dev         # pytest, ruff, import-linter
uv run pytest               # unit / integration / e2e
uv run lint-imports         # enforces: genie.* must not import applications.*
```
