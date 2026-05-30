# act2 — Multi-Agent Travel Assistant

A FastAPI service that answers travel questions (weather + hotel recommendations)
by routing each user message through a small graph of cooperating LLM agents.
The agents call tools served by a separate **MCP** (Model Context Protocol)
server, persist conversations to **MongoDB**, and emit structured traces to
**MLflow**.

---

## What it does

A user sends a chat message like *"What's the weather in Paris?"* to `POST /chat`.
The request flows through:

1. **RouterAgent** — reads the conversation, extracts `{location, intent}` as
   JSON, and decides whether to delegate to the weather agent, the hotel agent,
   or ask the user to clarify.
2. **WeatherAgent / HotelAgent** — invokes the matching MCP tool
   (`get_weather` / `get_hotels`) and formats a human-readable reply.
3. The final answer plus the full message history is written back to MongoDB
   so the next request in the same `thread_id` continues the conversation.

Every agent step is auto-wrapped in an MLflow span, so you get a per-request
trace of routing decisions, LLM calls, tool calls, and tool results.

---

## Capabilities

- **Multi-agent orchestration** via LangGraph (`graph/graph_builder.py`) —
  conditional edges route the state between router → specialist agent → END.
- **MCP tool integration** — agents discover and call tools from any MCP
  server (`sse`, `stdio`, `websocket`, `streamable_http`); a sample server
  shipping `get_weather` and `get_hotels` lives in `mcp_server/weather_server.py`.
- **OpenAI / OpenAI-compatible LLMs** — the LLM client is `ChatOpenAI`, so you
  can point `OPENAI_BASE_URL` at Azure, Groq, Together, or a local
  OpenAI-compatible endpoint without code changes.
- **Persistent memory** — MongoDB stores short-term conversation history
  (24-hour TTL) and long-term per-user facts that are injected into the
  system prompt on each turn.
- **Observability** — MLflow tracing + JSON logging via the `Observable`
  mix-in; spans capture inputs, outputs, exceptions, and named events.
- **Pluggable permissions** — tools can be filtered per agent / per user role
  before being bound to the LLM (`baseagent/permissions.py`).
- **Composable BaseAgent** — `BaseAgent` is a single class wired from three
  collaborators (`LLMClient`, `MCPClient`, `AgentMemory`). Subclasses set
  `system_prompt` and `tool_names`, then either override `run()` or call the
  one-shot `answer_with_tool()` template.

---

## HTTP API

| Method | Path                  | Purpose                                      |
| ------ | --------------------- | -------------------------------------------- |
| POST   | `/chat`               | Send a user message; returns the agent reply |
| GET    | `/health`             | Liveness probe                               |
| GET    | `/state/{thread_id}`  | Inspect the LangGraph checkpoint for a thread |

`POST /chat` body:

```json
{ "message": "Hotels in Tokyo?", "thread_id": "user-123" }
```

---

## Project layout

```
act2/
├─ main.py                 # FastAPI app, /chat handler, graph wiring
├─ state.py                # AgentState TypedDict (the shared blackboard)
├─ graph/
│  └─ graph_builder.py     # LangGraph nodes + conditional routing
├─ agents/
│  ├─ router_agent.py      # JSON-only intent + location extractor
│  ├─ weather_agent.py     # Calls MCP `get_weather`
│  └─ hotel_agent.py       # Calls MCP `get_hotels`
├─ baseagent/              # Reusable agent core (composition, not mixins)
│  ├─ base_agent.py        # BaseAgent: orchestration + state helpers
│  ├─ llm_client.py        # ChatOpenAI wrapper + tool execution
│  ├─ mcp_client.py        # MCP config + tool loading + result unwrapping
│  ├─ agent_memory.py      # Sliding-window trim + long-term fact persistence
│  ├─ permissions.py       # Role-based tool filtering (override per agent)
│  └─ events.py            # Log/event name constants
├─ mcp_server/
│  └─ weather_server.py    # Sample MCP server (get_weather, get_hotels)
├─ mcpconfig/mcp_config.py # MCP transport / server / agent config models
├─ memory/
│  ├─ memory.py            # LangGraph in-memory checkpointer + thread config
│  └─ mongo_store.py       # MongoDB-backed short-term + long-term store
├─ observability/
│  ├─ observable.py        # Observable base class + MLflow span auto-wrapping
│  ├─ mlflow_setup.py      # MLflow init
│  └─ logging.py           # JSON logger + MLflow span log handler
├─ frontend/               # Static UI mounted at /
├─ run.bat / kill.bat      # Windows launch + shutdown scripts
└─ requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.11+
- A running MongoDB instance (defaults to `mongodb://localhost:27017`)
- An OpenAI API key (or any OpenAI-compatible endpoint)
- Optional: an MLflow tracking server for persisted traces

### Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configure

```powershell
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY, MONGODB_URI, MLFLOW_TRACKING_URI, etc.
```

Key environment variables:

| Variable               | Purpose                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| `OPENAI_API_KEY`       | Required. API key for the LLM.                                         |
| `OPENAI_MODEL`         | Model name (default `gpt-4o-mini`).                                    |
| `OPENAI_BASE_URL`      | Optional override for OpenAI-compatible providers.                     |
| `MCP_SERVER_URL`       | MCP server URL. Leave empty to disable MCP tool loading entirely.      |
| `MCP_TRANSPORT`        | `sse` \| `stdio` \| `websocket` \| `streamable_http`.                  |
| `MCP_AUTH_TOKEN`       | Optional bearer token sent as `Authorization` to the MCP server.       |
| `MONGODB_URI`          | MongoDB connection string.                                             |
| `MLFLOW_TRACKING_URI`  | MLflow server URL. Without it, tracing is a no-op.                     |

### Run

On Windows, `run.bat` launches both the MCP server and the FastAPI app in
separate windows:

```powershell
.\run.bat
```

Or manually:

```powershell
# Terminal 1 — MCP tool server on :8001
python -m mcp_server.weather_server

# Terminal 2 — FastAPI app on :8000
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open <http://127.0.0.1:8000> for the static frontend, or POST directly to
`/chat`.

### Stop

```powershell
.\kill.bat
```

---

## Extending

**Add a new specialist agent**

1. Create `agents/my_agent.py` inheriting from `BaseAgent`.
2. Set `system_prompt` and `tool_names = ["my_mcp_tool"]`.
3. Implement `run(state)` — or call `self.answer_with_tool(...)` for the
   one-shot template used by `WeatherAgent` / `HotelAgent`.
4. Register the node and a routing branch in `graph/graph_builder.py`.

**Add a new MCP tool**

Add an `@mcp.tool()`-decorated function to `mcp_server/weather_server.py`
(or stand up a separate MCP server and point `MCP_SERVER_URL` at it). The
agent declaring it in `tool_names` will pick it up at startup.

**Use a different LLM provider**

Set `OPENAI_BASE_URL` and (if needed) `OPENAI_MODEL` — any OpenAI-compatible
endpoint works without code changes.
