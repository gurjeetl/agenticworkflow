# Genie Platform

A multi-agent **platform** built on **FastAPI** + **LangGraph**. Genie provides the shared
kernel — a router → planner → synthesizer workflow plus reusable capabilities (MCP tool
connectivity, agent registry/discovery, agent-to-agent messaging, multi-layer memory,
content-safety guards, centralized database connections, and observability) — and lets
**applications across the organization build their own agents** by inheriting a single
`BaseAgent` class.

Teams don't fork the platform; they ship agents *on* it. Each agent is a small subclass that
declares its prompt, the tools it needs, and an `AgentMeta` describing its capabilities. It
runs as its own service, self-registers with the central Registry, and is discovered and
orchestrated at runtime — no change to the kernel required.

> The bundled `weather`, `outage`, and `rag` agents under `src/applications/demo/` are
> **samples** that demonstrate the patterns. They are not the product — your agents are.

> **Installing or running the platform? See [SETUP.md](SETUP.md)** for prerequisites,
> dependencies, configuration, and the launcher scripts.

---

## Contents

1. [Platform vs. application](#platform-vs-application)
2. [Architecture](#architecture)
3. [How a request flows](#how-a-request-flows)
4. [Getting started](#getting-started)
5. [Dependencies](#dependencies)
6. [Configuration](#configuration)
7. [Database connections](#database-connections)
8. [Building an agent](#building-an-agent)
9. [AgentMeta & the registry](#agentmeta--the-registry)
10. [Creating MCP tools & connecting an MCP server](#creating-mcp-tools--connecting-an-mcp-server)
11. [Agent-to-agent (A2A) communication](#agent-to-agent-a2a-communication)
12. [Sample agents](#sample-agents)
13. [MLflow observability](#mlflow-observability)
14. [HTTP API](#http-api-gateway-8000)
15. [Memory backends](#memory-backends)
16. [Tests & boundaries](#tests--boundaries)
17. [Repository layout](#repository-layout)

---

## Platform vs. application

The codebase is split along one boundary, enforced by `import-linter`
(**the platform kernel may never import an application**):

- **`src/genie/`** — the **platform kernel**: the workflow (router → … → synthesizer) plus the
  capabilities every agent gets for free (LLM client, MCP connectivity, registry, A2A, memory
  stores, guards, database connections, observability, config).
- **`src/applications/`** — **domain agents** built *on* the platform. Each inherits
  `genie.agents.base.BaseAgent`, declares an `AgentMeta`, and runs as its own A2A service. The
  bundled `demo` application ships three samples: `weather`, `outage`, and `rag`.

Adding a capability to an application never requires touching the kernel. The boundary is a
CI-enforced contract (`uv run lint-imports`): `genie.*` must not import `applications.*`.

---

## Architecture

One prompt enters the gateway and flows through a LangGraph state machine. An **Input Guard**
screens it; a **Router** triages intent; a **Planner** turns it into a directed acyclic graph
(DAG) of subtasks; an **Orchestrator** computes dependency waves; an **Executor** runs each
wave by calling agents over A2A; a **Completion Gate** decides whether to re-plan or proceed;
a **Synthesizer** composes one answer; an **Output Guard** screens the reply. Agents run as
**independent A2A services** that self-register with the standalone **Registry**, and every
component emits **MLflow** spans that the companion **`/trace.html`** UI animates step-by-step.

```
POST /chat
   │
   ▼
START ─▶ Input Guard ─▶ Router ─┬─(fast)─────────────────────────▶ Executor ─┐
                                ├─(chitchat)──────────────────────▶ Synthesizer
                                └─(plan)▶ Planner ▶ Orchestrator ▶ Executor ▶ Gate ─┬─▶ Synthesizer ─▶ Output Guard ─▶ END
                                                                                    └─▶ Planner   (re-plan loop, budget 3)
```

Guards and the Router are optional (toggled by config); when off they are omitted from the
graph entirely. The capabilities the kernel provides:

| Capability | Where | Notes |
| --- | --- | --- |
| **MCP connectivity** | `genie/mcp/` | `BaseAgent` loads MCP tools from config; a single server via `MCP_SERVER_URL` or named servers via the YAML `mcp_services` block. Results normalized to `MCPToolResult`. |
| **Registry / discovery** | `genie/registry/`, `services/registry/` | Self-registration + heartbeat + TTL liveness, backed by MongoDB. |
| **A2A messaging** | `genie/a2a/` | JSON-RPC `message/send`, spec-compliant Agent Cards. |
| **Database connections** | `genie/platform/` | Centralized, reusable connectors for Mongo, Redis, Milvus, Postgres, SQL Server. |
| **Memory** | `genie/memory/` | Sessions, durable conversations, facts, commits, blackboard, vectors. |
| **Content guards** | `genie/security/` | Input/output guard (llm-guard); master on/off switch. |
| **Observability** | `genie/observability/` | MLflow spans auto-wrapped by `Observable`. |
| **Config** | `genie/platform/config.py` | Central `Settings` (pydantic-settings + YAML). |

### Runtime topology

The single most important architectural decision: **agents do not run in-process.** The
orchestration gateway imports no agent classes. Each agent is its own service that
self-registers with the Registry on startup; the Planner discovers them from the Registry's
live capability menu, and the Executor invokes them over A2A. Adding an agent requires no change
to the kernel — it just starts up and registers.

```
┌────────────────────────────────────────────────────────────────────────┐
│  GATEWAY  (app:app, :8000)   FastAPI + LangGraph pipeline + frontend     │
│  START → input_guard → router ┬─► planner → orchestrator → executor      │
│                               ├─(fast)──────────────────► executor       │
│                               └─(chitchat)────────────► synthesizer       │
│                            … → gate → synthesizer → output_guard → END    │
└───────────┬──────────────────────────────────────┬─────────────────────┘
   discovery │ (HTTP GET /agents)        A2A JSON-RPC │ message/send → POST {endpoint}/a2a
            ▼                                          ▼
┌──────────────────────────┐         ┌────────────────────────────────────┐
│ REGISTRY SERVICE  :8002  │◄────────┤ AGENT SERVICES (one process each)   │
│ register · heartbeat ·   │ register│  weather :8010 · outage :8011 · …    │──┐ MCP tools
│ TTL liveness (MongoDB)   │   + hb   │  BaseAgent + run_agent harness      │  │
└──────────────────────────┘         │  POST /a2a · /.well-known/agent.json │  ▼
            │                        └────────────────────────────────────┘ ┌─────────────────┐
            ▼                                                                │ MCP SERVER :8001│
┌────────────────────────────────────────────────────────────────────────┐ │ get_weather, …  │
│  STATE & MEMORY   MongoDB (sessions·conversations·facts·commits; +       │ └─────────────────┘
│  registry) · Redis (per-run blackboard) · Milvus (semantic long-term) ·  │
│  LangGraph checkpointer (per-thread graph state)                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## How a request flows

Each node lives under `src/genie/application/nodes/` (guards under `src/genie/security/`) and
is auto-wrapped in an MLflow span.

1. **Input Guard** (`security/guards.py`) screens the prompt (prompt-injection, toxicity,
   banned topics) and redacts PII/secrets. A blocking finding short-circuits to a safe refusal.
2. **Router** (`nodes/router.py`) does cheap intent triage — a regex heuristic + a local
   embedding classifier (`_router_intent.py`) + an LLM fallback — and routes to `fast` (one
   obvious agent → straight to the Executor), `chitchat` (→ Synthesizer), or `plan` (→ full
   pipeline). It **fails open**: any registry/LLM/parse error downgrades to `plan`.
3. **Planner** (`nodes/planner.py`) discovers live agents from the Registry, recalls session
   memory (facts + semantic), and emits a JSON DAG
   `{"subtasks":[{"id","agent_id","args","depends_on","sla_ms"}]}`, validating each subtask's
   args against the agent's `input_schema` (unknown agents / invalid args are dropped).
4. **Orchestrator** (`nodes/orchestrator.py`) computes dependency **waves** with Kahn's
   algorithm; independent tasks share a wave.
5. **Executor** (`nodes/executor.py`) runs each wave concurrently, invoking each chosen agent
   over **A2A JSON-RPC** (`message/send` to `POST {endpoint}/a2a`). It resolves
   `${task_id.path}` argument references (so one task can consume another's output), and every
   result — success or error — lands on a shared **blackboard**.
6. **Completion Gate** (`nodes/completion_gate.py`) inspects the blackboard: all subtasks
   satisfied? errors? re-plan budget left (`max_replans`, default **3**)? Routes back to the
   Planner or onward.
7. **Synthesizer** (`nodes/synthesizer.py`) merges the blackboard into one answer (marking
   `[PARTIAL]` where agents errored) and writes durable facts/commits + long-term memory.
8. **Output Guard** screens the final answer before it reaches the user.

**The re-plan loop** is the only cycle in the graph. When a wave fails or a subtask is missing
and budget remains, the Gate routes back to the Planner with the failure reason; successful
tasks are seeded from the prior attempt's blackboard snapshot so they don't re-run. Out of
budget, the Synthesizer returns a `[PARTIAL]` answer rather than failing.

**State & blackboard** (`application/state.py`, `blackboard.py`): the graph threads a single
typed `AgentState`. Each node returns a *new* state via a `patch(state, **changes)` helper —
nodes never mutate in place; `messages` is the one reducer field (it accumulates), everything
else is overwrite-on-write. The blackboard holds per-task results keyed
`bb:{thread_id}:{run_id}:{task_id}`, mirrored to Redis (best-effort) and surfaced via
`GET /blackboard/...`. A LangGraph checkpointer (`checkpointer.py`) scopes graph state per
`thread_id` across turns.

### Worked example

Request: **"Weather in Tokyo and the top outages."**

1. **Planner** discovers `weather` + `outage` from the Registry, renders the capability menu,
   and the LLM returns two independent subtasks:
   ```json
   {"subtasks":[
     {"id":"t1","agent_id":"weather","args":{"location":"tokyo"},"depends_on":[]},
     {"id":"t2","agent_id":"outage","args":{},"depends_on":[]}]}
   ```
2. **Orchestrator** sees no dependencies → a single wave: `[["t1","t2"]]`.
3. **Executor** fires both agents concurrently over A2A (`POST {endpoint}/a2a`): the weather
   agent calls the MCP `get_weather` tool, the outage agent calls `list_outage_ids`. Both
   results land on the blackboard.
4. **Gate** sees both satisfied, no errors → synthesize.
5. **Synthesizer** merges into one answer: *"The current weather in Tokyo is humid… Additionally,
   there are 199 total outages, with the top 5 highlighted."*

Had the outage agent timed out, the Gate would route back to the Planner (if budget remained),
or the Synthesizer would mark that section `[PARTIAL]`. To **chain** dependent agents, a subtask
references an upstream output in its args, e.g. `"outage_id": "${t1.view.items.0.id}"`, which the
Executor resolves from the blackboard before the call.

---

## Getting started

**Full install, configuration, and run instructions live in [SETUP.md](SETUP.md).** In short:

```powershell
uv sync --extra dev                                            # install (incl. test/lint tools)
Copy-Item config\local.yaml.example config\local.yaml          # set openai_api_key, etc.

# Run the platform WITHOUT the sample agents (bring your own):
powershell -ExecutionPolicy Bypass -File scripts\run-platform.ps1

# …or the WHOLE stack WITH the bundled sample agents:
powershell -ExecutionPolicy Bypass -File scripts\run-full.ps1
```

Then open <http://127.0.0.1:8000> (chat UI) and <http://127.0.0.1:8000/trace.html> (tracer).
The stack is multi-process — gateway :8000, MLflow :5000, MCP :8001, registry :8002, RAG
service :8003, sample agents :8010–:8012. See [SETUP.md](SETUP.md) for the per-process commands,
the other launchers (`run-dev`, `run-prod`, `run-all`), and troubleshooting.

---

## Dependencies

Declared in `pyproject.toml` (`requires-python = ">=3.13"`):

- **Web / orchestration**: `fastapi`, `uvicorn[standard]`, `langchain`, `langgraph`,
  `langchain-mcp-adapters`, `langchain-openai`, `langchain-core`, `python-dotenv`
- **Config**: `pydantic-settings`, `PyYAML`
- **Persistence drivers**:
  - `motor` (async MongoDB; brings `pymongo` for sync) — **required backend**
  - `redis>=4.2`, `pymilvus[milvus_lite]>=2.4` — optional stores
  - `psycopg[binary]>=3.1` + `psycopg_pool>=3.2` (PostgreSQL, sync + async + pooling)
  - `pyodbc>=5.0` + `aioodbc>=0.5` (SQL Server, sync + async) — needs the OS **"ODBC Driver 18
    for SQL Server"**
- **Security**: `llm-guard>=0.3.15` (input/output content guard)
- **Router**: `sentence-transformers>=2.7` (local CPU intent classifier)
- **Observability**: `mlflow>=2.16`, `psycopg2-binary` (used only by the MLflow launcher)
- **RAG boundary**: `genie-rag-contracts` (editable local package under `[tool.uv.sources]`)
- **Dev extra**: `pytest`, `pytest-asyncio`, `ruff`, `import-linter`

> **Vendored llm-guard**: upstream caps `requires-python` below 3.13, so the platform vendors
> its source under `vendor/llm-guard/` with that cap raised (the only change). Models are still
> fetched from HuggingFace at runtime. Drop the override once upstream lifts the cap.

---

## Configuration

Configuration is centralized in `genie.platform.config.Settings` (pydantic-settings). It
resolves each key in priority order (**first wins**):

1. `config/local.yaml` (gitignored — secrets & machine overrides)
2. `config/default.yaml` (committed canonical config; or `$GENIE_CONFIG_FILE`)
3. environment variables / `.env`
4. built-in field defaults

**YAML is authoritative over the environment** — an env var only fills a key no YAML sets.
`env_prefix=""`, so a field maps to its UPPER_SNAKE name (`mongodb_uri` ← `MONGODB_URI`).
**Nested** structures that env vars can't express — named MCP servers under `mcp_services`,
LLM backends under `llm_services` — live only in YAML.

Key settings (see `config.py` for the full list):

| Variable | Default | Purpose |
| --- | --- | --- |
| `openai_api_key` / `openai_base_url` / `openai_model` | — / — / `gpt-4o-mini` | LLM credentials + model (any OpenAI-compatible endpoint). |
| `mongodb_uri` / `mongodb_db` | `mongodb://localhost:27017` / `agent_memory` | Primary datastore (required; also backs the registry). |
| `redis_url` | `None` | Optional; enables the blackboard hot mirror. |
| `milvus_uri` / `milvus_db_path` / `milvus_token` | `None` | Optional; semantic long-term memory. |
| `postgres_dsn` | `None` | Optional; libpq URI (env `POSTGRES_DSN`). |
| `sqlserver_dsn` | `None` | Optional; full ODBC connection string (env `SQLSERVER_DSN`). |
| `mcp_server_url` / `mcp_transport` / `mcp_auth_token` | `None` / `sse` / `""` | Single MCP server (or use `mcp_services` in YAML). |
| `registry_url` / `registry_ttl_seconds` / `registry_heartbeat_seconds` | `http://127.0.0.1:8002` / `90` / `30` | Discovery + liveness window. |
| `agent_port` / `agent_advertise_host` / `agent_advertise_port` | `8010` / `None` / `None` | Agent service bind + advertised endpoint. |
| `mlflow_tracking_uri` / `mlflow_experiment_name` | `None` / `base-agent-framework` | Tracing target (no-op when unset). |
| `llm_guard_enabled` | `True` | Master switch for the content guard. |
| `router_enabled` / `router_intent_backend` | `False` / `embedding` | Intent triage on/off + backend (`embedding` or `llm`). |

---

## Database connections

All datastore access goes through **one reusable connection layer** under
`src/genie/platform/` — agents, tools, and kernel nodes call an accessor instead of
constructing their own client. Connection strings come from `Settings` (YAML-first), so a
backend is configured once and reused everywhere.

| Backend | Module | Accessors | Required? |
| --- | --- | --- | --- |
| MongoDB | `platform/mongo.py` | `get_sync_mongo_db()`, `get_async_mongo_db()` (+ client variants) | **Yes** |
| Redis | `platform/redis.py` | `get_async_redis_client()` (per-loop), `get_sync_redis_client()` | No |
| Milvus | `platform/milvus.py` | `get_milvus_client()` (sync only) | No |
| PostgreSQL | `platform/postgres.py` | `get_pg_connection()`, `get_async_pg_connection()`, `postgres_healthcheck()` | No |
| SQL Server | `platform/sqlserver.py` | `get_sqlserver_connection()`, `get_async_sqlserver_connection()`, `sqlserver_healthcheck()` | No |

Shutdown is centralized: `genie.platform.db.close_all_connections()` closes whatever this
process opened (called from the gateway and registry lifespans).

**Sync vs. async — the one rule to remember:**

- Use the **sync** accessors (`pymongo`, `psycopg`, `pyodbc`) from **LangGraph nodes, agents,
  and MCP tools**. These run synchronously, often on transient event loops.
- Use the **async** accessors (`motor`, `redis.asyncio`, async psycopg/aioodbc) only from the
  **gateway's event loop**.

An async (motor/redis) client is bound to the loop that created it and raises *"attached to a
different loop"* if shared across loops — hence the split (Redis additionally caches one async
client per loop). Optional backends **fail open**: their accessor returns `None` (Redis/Milvus)
or raises a clear *"not configured"* error (Postgres/SQL Server) when the DSN is unset; Mongo
is required.

Grabbing a connection in a tool or agent is a one-liner:

```python
from genie.platform.mongo import get_sync_mongo_db
from genie.platform.postgres import get_pg_connection

# MongoDB — collection handle off the shared sync client
doc = get_sync_mongo_db()["customers"].find_one({"_id": customer_id})

# PostgreSQL — connection from the shared pool (context-managed)
with get_pg_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM customers WHERE id = %s", (customer_id,))
        row = cur.fetchone()
```

> **The data-access seam pattern.** The sample MCP server reads its data through
> `services/mcp/_repository.py` rather than inline. When you move a tool's data from a sample
> file to a real database, you change only the repository function bodies (swap them for the
> `get_*_connection()` calls above) — the tools, their output schemas, and the agents stay
> untouched.

---

## Building an agent

An application agent is a `BaseAgent` subclass plus an `AgentMeta`. The platform supplies the
LLM client, MCP tool loading, and working memory; the agent declares which tools it wants
(`tool_names`) and what to do (`run`).

### `BaseAgent` API (`src/genie/agents/base.py`)

- `system_prompt: str` — the system prompt for LLM calls.
- `tool_names: list[str] | None` — MCP tool loading: `None` = load all permitted tools, `[]` =
  skip MCP entirely (pure-LLM agent), `[...]` = load only the named tools. Tools are loaded
  once at construction from the configured MCP server(s).
- `run(state) -> AgentState` — the entry point the executor/graph calls. Override it, or rely
  on the default (one LLM call when tool-less, else an LLM↔tool loop).
- `answer_with(state, work, **trace)` — run a zero-arg callable that returns `text` or
  `(text, view)`; handles bookkeeping, error capture, and the final state mutation.
- `answer_with_tool(state, tool_name, args, format_text, **trace)` — call one MCP tool and
  format its result. **`format_text` receives an `MCPToolResult`** (see below) and returns
  `text` or `(text, view)`.
- `call_mcp_tool_structured(name, args) -> MCPToolResult` — invoke one tool, get the normalized
  result; raises `LookupError` if the tool is missing or reported an error.
- `call_mcp_tool(name, args) -> str` — convenience wrapper returning just `.text`.
- `call_llm(messages) -> str` — one-shot LLM call (no tool loop).
- `call_peer(agent_id, args, context=None, *, sla_ms=10000) -> str` — delegate to another agent
  over A2A (see [A2A](#agent-to-agent-a2a-communication)).

`MCPToolResult` (`genie/mcp/client.py`) is the spec-aligned view of a tool call:
`.text` (human/LLM-readable), `.structured` (the parsed `structuredContent` — a dict/list),
`.blocks` (non-text content as base64-free references), `.is_error`.

### Minimal subclass

```python
from genie.agents.base import BaseAgent
from genie.registry import AgentMeta, FieldSpec, Skill
from genie.application.state import AgentState


class WeatherAgent(BaseAgent):
    system_prompt = "You are a helpful weather reporter."
    tool_names = ["get_weather"]          # only this MCP tool is bound

    def run(self, state: AgentState) -> AgentState:
        city = (state.get("location") or "").lower().strip()
        return self.answer_with_tool(
            state,
            tool_name="get_weather",
            args={"city": city},
            # format_text receives an MCPToolResult; read its parsed .structured
            format_text=lambda res: (
                f"Here's the weather for {city.title()}: "
                f"{(res.structured or {}).get('report', res.text)}"
            ),
        )


META = AgentMeta(
    agent_id="weather",
    capability_tags=["weather", "forecast", "city"],   # what the Router/Planner match on
    description="Reports current weather conditions for a named city.",
    input_schema={"location": FieldSpec(type="string", required=True, description="City name.")},
    output_schema={"text": FieldSpec(type="string", persist=True)},
    sla_ms=4000,
)

if __name__ == "__main__":                              # run as its own A2A service
    from genie.agents.server import run_agent
    run_agent(WeatherAgent, META)
```

Run it on a free port (`$env:AGENT_PORT="8010"`) and it self-registers; the Planner discovers
it on the next request. No kernel change required.

---

## AgentMeta & the registry

`AgentMeta` (`src/genie/registry/agent_meta.py`) is the contract an agent advertises. It is
served both as the registry record and — projected identically — as the A2A Agent Card.

| Field | Default | Meaning |
| --- | --- | --- |
| `agent_id` | *(required)* | Unique agent identifier. |
| `version` | `"1.0.0"` | Agent version (stamped onto planned subtasks). |
| `capability_tags` | `[]` | Tags the Router/Planner match intent against. |
| `description` | `""` | Human-readable summary (used in the capability menu). |
| `input_schema` | `{}` | `dict[str, FieldSpec]` of expected args. |
| `output_schema` | `{}` | `dict[str, FieldSpec]` of outputs. |
| `skills` | `[]` | A2A `Skill`s; **auto-derived** from tags + description + input_schema when empty, so the registry record and Agent Card never drift. |
| `sla_ms` | `10000` | Target execution budget (ms). |
| `transport` | `"json-rpc"` | A2A transport. |
| `status` | `"active"` | `active` / `deprecated`. |
| `endpoint` | `None` | Base URL the Executor POSTs to (`/a2a` appended); stamped when run as a service. |
| `instance_id` | `None` | Unique per process; the registry assigns a uuid4 if absent. |
| `last_heartbeat` / `registered_at` | `None` | **Server-owned** liveness fields, stamped by the registry. |

`FieldSpec(type, required, description, persist)` — `type` is a JSON-Schema scalar; `persist=True`
makes the Synthesizer commit that output field to MongoDB. `Skill(id, name, description, tags,
examples)` mirrors the A2A `AgentSkill`.

### How an agent is saved in the registry

The `run_agent(AgentClass, META)` harness (`src/genie/agents/server.py`) handles the full
lifecycle — your agent's `__main__` just calls it:

1. **On startup** the harness stamps `META` with the advertised `endpoint`
   (`agent_advertise_host/port` or `agent_host`/`AGENT_PORT`) and a fresh `instance_id`, then
   **POSTs `/register`** to the registry. The registry persists the record (MongoDB) and
   returns the TTL + heartbeat interval.
2. A background loop **POSTs `/heartbeat`** every `registry_heartbeat_seconds` (default 30). If
   the registry reports the instance as unknown (e.g. after a TTL sweep), the agent
   re-registers automatically.
3. Records expire `registry_ttl_seconds` (default 90) after the last heartbeat — enforced by a
   MongoDB TTL index *and* a freshness filter on reads, so a crashed agent disappears from
   discovery on its own. On graceful shutdown the harness **POSTs `/deregister`**.

The harness also serves, per agent: `GET /health`, `GET /.well-known/agent.json` (the A2A
Agent Card), and `POST /a2a` (the JSON-RPC endpoint, optionally Bearer-protected via
`agent_invoke_token`).

**Registry service** (`services/registry/server.py`, :8002): `POST /register`,
`POST /heartbeat`, `POST /deregister`, `GET /agents[?agent_id=&tag=]`, `GET /agents/{agent_id}`,
`GET /health` (all Bearer-protected when `registry_auth_token` is set). Inside the kernel, the
Router/Planner/Executor discover agents through `get_registry_client().list_active()` (a thin
HTTP client with a ~5s in-process cache).

**Minimum metadata to be usable:** an `agent_id`, a reachable `endpoint` (supplied by the
harness), and enough `capability_tags` + `description` + `input_schema` for the Router/Planner
to match intent and build valid args.

---

## Creating MCP tools & connecting an MCP server

Tools are exposed by an **MCP server** and consumed by agents through the platform's MCP client.

### Defining a tool (server side)

The sample server is `services/mcp/genie_mcp_server.py` (FastMCP over SSE on :8001). A tool is
an `@mcp.tool()`-decorated function that **returns a Pydantic model** (so FastMCP advertises an
`outputSchema` and ships `structuredContent` — clients receive parsed objects, not stringified
JSON) and **raises `ToolError`** to signal failure (rendered as MCP `isError`, never an error
payload):

```python
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel

mcp = FastMCP("genie-mcp-server", host="127.0.0.1", port=8001)
_READ_ONLY = ToolAnnotations(readOnlyHint=True)


class WeatherReport(BaseModel):
    city: str
    report: str


@mcp.tool(annotations=_READ_ONLY)
def get_weather(city: str) -> WeatherReport:
    """Return the current weather report for the given city."""
    key = (city or "").strip().lower()
    report = WEATHER_DATA.get(key)
    if report is None:
        raise ToolError(f"No weather data available for '{city}'.")
    return WeatherReport(city=key, report=report)
```

Tool data should go through a **data-access seam** (`services/mcp/_repository.py`) so swapping
sample data for a database touches only the repository — e.g. a DB-backed tool:

```python
from genie.platform.mongo import get_sync_mongo_db

@mcp.tool(annotations=_READ_ONLY)
def get_customer(customer_id: str) -> Customer:
    doc = get_sync_mongo_db()["customers"].find_one({"_id": customer_id})
    if doc is None:
        raise ToolError(f"No customer {customer_id}.")
    return Customer(**doc)
```

The sample server ships: `get_weather`, `get_outage_report_summary`, `list_outage_ids`,
`get_outage_metadata`, `get_outage_analysis_summary`, `get_outage_attribute_analysis`,
`get_linked_outages`, and `search_docs` (BM25 over the repo's markdown docs; backs the RAG
sample agent).

### Connecting an MCP server (client side)

Agents load tools via the platform MCP client (`genie/mcp/client.py`, built on
`langchain_mcp_adapters`). Point it at a server two ways:

- **Single flat server** — set `mcp_server_url` (+ `mcp_transport`, `mcp_auth_token`,
  `mcp_timeout`). Transports: `sse` | `stdio` | `websocket` | `streamable_http`.
- **Named servers** — declare an `mcp_services` block in YAML (multiple servers, each with its
  own url/transport/token).

At construction, an agent connects, narrows to its `tool_names`, applies
`filter_tools_by_permission` (`genie/mcp/permissions.py` — override for RBAC), and binds the
tools. Tool results are normalized to `MCPToolResult` so structured output reaches your agent
as a parsed object.

---

## Agent-to-agent (A2A) communication

Agents talk over **A2A JSON-RPC** (`message/send` to `POST {endpoint}/a2a`), discovering each
other through the Registry — they never import one another.

- **The Executor** invokes each planned agent this way, threading invocation context
  (`task_id`, `run_id`, `thread_id`, the current `blackboard`, `sla_ms`) in the message
  metadata and the args in a `DataPart`. The reply is an agent-role `Message`: a `TextPart`
  with the answer plus an optional `DataPart` carrying a structured `view`.
- **`BaseAgent.call_peer`** lets an agent delegate mid-run:

```python
# inside an agent's run(): fan work out to another agent
reply_text = self.call_peer("weather", {"location": "Paris"})
```

`A2AClient.send(agent_id, args, context, sla_ms)` (`genie/a2a/client.py`) resolves the peer's
endpoint via the registry and posts the JSON-RPC request; `genie/a2a/types.py` defines the
`Message` / `TextPart` / `DataPart` shapes and `get_text()` helper. Each agent serves a
spec-compliant Agent Card at `/.well-known/agent.json` for external A2A interoperability.

**Chaining agents in a plan:** the Planner can wire one task's output into another's args using
`${task_id.path}` references that the Executor resolves against the blackboard, e.g.
`"outage_id": "${t1.view.items.0.id}"` feeds the first listed outage's id into a downstream task.

---

## Sample agents

These ship under `src/applications/demo/` and illustrate the common patterns. They are
**samples** — copy the shape, not the domain.

**Single tool, structured result** — `weather` (`demo/weather/agent.py`): see the
[minimal subclass](#minimal-subclass) above. Calls one MCP tool and reads `res.structured`.

**Multiple tools + a structured view** — `outage` (`demo/outage/agent.py`): uses
`call_mcp_tool_structured` directly and returns a `(text, view)` tuple the frontend renders:

```python
class OutageAgent(BaseAgent):
    system_prompt = "You are a grid-outage analyst summarizing outage reports."
    tool_names = ["list_outage_ids", "get_outage_metadata", "get_outage_analysis_summary"]

    def _list_view(self):
        data = self.call_mcp_tool_structured("list_outage_ids", {}).structured or {}
        items = data.get("items", [])
        if not items:
            return "No outages found in the current report."
        text = f"Top {len(items)} outages (of {data.get('total')} total)."
        return text, {"type": "outage_list", "total": data.get("total"), "items": items}

    def run(self, state):
        outage_id = state.get("outage_id")
        if outage_id is not None:
            return self.answer_with(state, lambda: self._detail_view(int(outage_id)),
                                    source="mcp:outage_detail")
        return self.answer_with(state, self._list_view, source="mcp:outage_list")
```

**Pure-LLM / retrieval** — `rag` (`demo/rag/agent.py`): retrieves doc chunks via the
`search_docs` MCP tool, then composes a grounded answer with `call_llm`, attaching a sources
`view`. (An agent with `tool_names = []` skips MCP entirely and is pure-LLM.)

---

## MLflow observability

Every Router, Planner, Orchestrator, Gate, Synthesizer, and Agent invocation is auto-wrapped in
an MLflow span by the `Observable` base class (`genie/observability/observable.py`). A class
declares `_traced_methods` (and a `_span_type`); each listed method becomes a span capturing the
per-request `run_id`, `agent_id`/`version`, wave/retry counts, and structured events for tool
calls, blackboard writes, and gate decisions. Structured log records emitted inside a span are
attached to it as events.

MLflow is initialized once at startup via `init_mlflow` (`genie/observability/`), which reads
`mlflow_tracking_uri` and `mlflow_experiment_name` and enables `mlflow.langchain.autolog`. When
`mlflow_tracking_uri` is unset, tracing **degrades to a no-op — it never crashes the app**. Point
it at a tracking server to persist traces, or a local `sqlite:///mlflow_local.db` store.

The `run-*.ps1` launchers start an MLflow server on **:5000** backed by the
`mlflow_backend_store_uri` PostgreSQL DSN. The gateway also exposes `POST /chat/trace`, which
runs the graph in streaming mode and returns every node update; `frontend/trace.html` animates
that step-by-step (recommended starting point for understanding a run).

---

## HTTP API (gateway, :8000)

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/chat` | Send a prompt; returns `{response, view}`. |
| POST | `/chat/trace` | Same pipeline, but returns every node update for the tracer UI. |
| GET | `/registry` | Live discovery: every registered agent + schema, endpoint, last heartbeat. |
| GET | `/state/{thread_id}` | LangGraph checkpoint snapshot for a thread. |
| GET | `/blackboard/{thread_id}/{run_id}` | Redis-mirrored blackboard entries for a run. |
| GET | `/conversations` · `/conversations/{thread_id}` | List / resume past conversations. |
| DELETE | `/conversations/{thread_id}` | Delete a conversation. |
| GET | `/health` | Liveness probe. |

`POST /chat` body:

```json
{ "message": "Show me the top 5 outages", "thread_id": "user-123" }
```

---

## Memory backends

| Store | Purpose | Required? |
| --- | --- | --- |
| **MongoDB** | Session messages + durable conversations + per-thread facts + durable commits; also backs the registry (TTL liveness). | **Yes** |
| **Redis** | Hot blackboard mirror keyed `bb:{thread_id}:{run_id}:{task_id}`; read via `GET /blackboard/...`. | No — no-ops when `redis_url` unset. |
| **Milvus** | Semantic long-term memory (embeddings the Planner recalls from). | No — no-ops when `milvus_uri`/`milvus_db_path` unset. |

Redis and Milvus are optional; the platform runs end-to-end with just MongoDB. PostgreSQL and
SQL Server are available through the [connection layer](#database-connections) for your own
agents/tools — they are not required by the kernel.

---

## Tests & boundaries

```powershell
uv sync --extra dev         # pytest, pytest-asyncio, ruff, import-linter
uv run pytest               # unit / integration / e2e
uv run lint-imports         # enforces: genie.* must not import applications.*
```

---

## Repository layout

```
genie-platform/
├─ src/
│  ├─ app.py                          # gateway entry: app = create_app()  (uvicorn app:app)
│  ├─ genie/                          # ── PLATFORM KERNEL ──
│  │  ├─ platform/                    # config.py (Settings) + DB connections:
│  │  │                               #   mongo, redis, milvus, postgres, sqlserver, db (close_all)
│  │  ├─ agents/                      # base.py (BaseAgent), server.py (run_agent A2A harness), memory
│  │  ├─ application/                 # graph.py, state.py, blackboard.py, checkpointer.py
│  │  │  └─ nodes/                    # router, planner, orchestrator, executor, completion_gate, synthesizer
│  │  ├─ interface/                   # bootstrap.py (create_app) + routers/{chat,health,state,registry,conversations}
│  │  ├─ llm/                         # ChatOpenAI client wrapper
│  │  ├─ mcp/                         # MCP client, config, permission filter, MCPToolResult
│  │  ├─ registry/                    # AgentMeta/FieldSpec/Skill, MongoDB store, registry client
│  │  ├─ a2a/                         # JSON-RPC client, Agent Card, types
│  │  ├─ memory/                      # mongo / facts / commit / redis / vector stores
│  │  ├─ security/                    # guards.py (graph nodes) + llm_guard.py
│  │  └─ observability/              # Observable, logging, MLflow setup
│  └─ applications/
│     └─ demo/                        # ── SAMPLE AGENTS ──
│        ├─ weather/agent.py · outage/agent.py · rag/agent.py
│        └─ providers.py              # launch manifest (AgentClass, META, port)
├─ services/
│  ├─ registry/server.py              # standalone Registry/Discovery service (:8002)
│  ├─ mcp/{genie_mcp_server,_repository,rag_index}.py   # standalone MCP tool server (:8001)
│  └─ rag/server.py                   # standalone RAG retrieval service (:8003)
├─ packages/genie-rag-contracts/      # shared wire contracts for the RAG service boundary
├─ config/{default,local.yaml.example,test}.yaml   # pydantic-settings YAML sources
├─ frontend/{index,trace}.html        # chat UI + step-by-step execution tracer
├─ tests/                             # unit / integration / e2e
├─ scripts/run-{dev,all,prod}.ps1     # launchers for the multi-process stack
└─ pyproject.toml                     # packaging + dependencies + import-linter contracts
```
