# BaseAgentFramework — Architecture

A multi-agent orchestration framework for building conversational AI workflows. A
single FastAPI app drives a [LangGraph](https://langchain-ai.github.io/langgraph/)
state machine that triages each request, plans work across **independently
deployed agent services**, executes that plan, guards the result, and synthesizes
one answer — with full MLflow tracing and a layered memory subsystem behind it.

> **Stack:** Python · FastAPI · LangGraph · OpenAI-compatible LLM · MongoDB ·
> Redis · Milvus · MLflow · llm-guard

---

## 1. Big picture

The system is split into three runtime tiers:

```
┌──────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATION APP  (main.py, port 8000)                               │
│  FastAPI + LangGraph pipeline + static frontend                        │
│                                                                        │
│   START → input_guard → router ┬─► planner → orchestrator → executor   │
│                                │                               │       │
│                                │                              gate ──┐  │
│                                │                               │     │  │
│                                ├──(fast)──────────────► executor     │  │
│                                └──(chitchat)───────────────────────┐ │  │
│                                                                    ▼ ▼  │
│                                              synthesizer → output_guard │
│                                                              → END      │
└──────────────────────────────────────────────────────────────────────┘
        │ discovery (HTTP)              │ A2A JSON-RPC message/send (HTTP)
        ▼                               ▼
┌─────────────────────────┐   ┌──────────────────────────────────────────┐
│ REGISTRY SERVICE        │   │ AGENT SERVICES (one process each)          │
│ registry.service :8002  │◄──┤  weather · outage · rag · ...              │
│ self-register/heartbeat │   │  each: BaseAgent + agent_server harness     │
│ discovery source-of-truth│  │  POST /a2a · GET /.well-known/agent.json    │
└─────────────────────────┘   └──────────────────────────────────────────┘
        │                                       │
        ▼                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STATE & MEMORY                                                        │
│  MongoDB (sessions, conversations, facts, commits) ·                   │
│  Redis (per-run blackboard, 1h TTL) · Milvus (semantic long-term)      │
│  LangGraph checkpointer (per-thread graph state)                       │
└──────────────────────────────────────────────────────────────────────┘
```

Key architectural decision: **agents do not run in-process.** Each agent is its
own service that self-registers with the Registry. The Planner discovers them
from the Registry's live capability menu, and the Executor invokes them over
A2A (agent-to-agent) JSON-RPC. Adding an agent requires no change to the
orchestration app — it just starts up and registers.

---

## 2. Request lifecycle

Entry point: [`POST /chat`](main.py#L109) in [main.py](main.py). The handler loads
prior session memory + durable facts, seeds an [`AgentState`](state.py), and calls
`graph.invoke(...)`. The compiled LangGraph
([graph/graph_builder.py](graph/graph_builder.py)) walks these nodes:

| # | Node | Module | Responsibility |
|---|------|--------|----------------|
| 1 | **input_guard** | [security/guards.py](security/guards.py) | Scan the user prompt (prompt-injection, toxicity, banned topics → **block**; PII/secrets → **redact**). Blocked ⇒ short-circuit to END with a safe refusal. |
| 2 | **router** | [router/router_agent.py](router/router_agent.py) | Cheap intent triage → one of three routes (below). |
| 3 | **planner** | [planner/planner_agent.py](planner/planner_agent.py) | Decompose the prompt into a **DAG of subtasks**, one per matched agent, using the Registry capability menu + recalled memory. |
| 4 | **orchestrator** | [orchestrator/orchestrator.py](orchestrator/orchestrator.py) | Topologically sort the DAG into **dependency waves** (Kahn's algorithm). Runs no agents. |
| 5 | **executor** | [orchestrator/executor.py](orchestrator/executor.py) | Run each wave concurrently, invoking agents over A2A, writing results to the shared **blackboard**. |
| 6 | **gate** | [gate/completion_gate.py](gate/completion_gate.py) | Are all subtasks present and error-free? If not (and budget remains) → **re-plan**; else → synthesize. |
| 7 | **synthesizer** | [synthesizer/synthesizer.py](synthesizer/synthesizer.py) | Merge blackboard entries into one user-facing answer; extract durable facts in the background. |
| 8 | **output_guard** | [security/guards.py](security/guards.py) | Scan the final answer before it reaches the user. |

### The three router routes
Decided in [`route_after_router`](graph/graph_builder.py#L30):

- **`fast`** — request maps to exactly one agent with fillable args. The Router
  builds a one-task plan + waves itself and jumps **straight to the Executor**,
  skipping the heavy Planner.
- **`chitchat`** — greeting / thanks / meta question. Skips to the Synthesizer,
  whose empty-plan path returns a clarification.
- **`plan`** — anything ambiguous or multi-intent. The **safe default**; the
  Router fails open to `plan` on any registry outage, LLM error, or low
  confidence, so it can only ever speed things up, never reduce capability.

A pre-LLM multi-intent check (regex + a local embedding intent classifier,
[router/intent_classifier.py](router/intent_classifier.py)) sends obviously
multi-agent prompts straight to `plan` without paying for the Router's LLM call.

### Re-plan loop
The **gate → planner** edge is the only cycle in the graph. When a wave fails or
a subtask is missing, the gate routes back to the planner (up to `max_replans`,
default 3). Successful tasks are seeded from the prior attempt's blackboard
snapshot so they don't re-run. Out of budget ⇒ the synthesizer returns a
**partial** answer rather than failing.

---

## 3. State

A single typed dict, [`AgentState`](state.py), threads through every node. Each
node returns a *new* state via the `patch(state, **changes)` helper
([baseagent/base_agent.py](baseagent/base_agent.py#L22)) — nodes never mutate in
place. `messages` is the one reducer field (`operator.add`), so message lists
accumulate across nodes; everything else is overwrite-on-write.

Notable groups: routing (`route`), planning (`plan`, `waves`, `agent_versions`),
execution (`blackboard`, `blackboard_snapshot`), re-plan control (`replan_count`,
`max_replans`, `partial`), guards (`guard_block`, `guard_input`, `guard_output`),
output (`final_output`, `view`, `is_complete`), and `db_ops` — per-node records of
real store operations surfaced to the trace UI.

---

## 4. Agents

### BaseAgent
[`BaseAgent`](baseagent/base_agent.py#L48) is the composed building block: an
`LLMClient` (OpenAI-compatible chat + tool binding), an `MCPClient` (loads tools
from an MCP server), and `AgentMemory` (message trimming + facts block). It
extends `Observable` so `run()` is automatically traced. Subclasses set a
`system_prompt` and `tool_names`, then either:

- override `run()` for bespoke logic (e.g. [WeatherAgent](agents/weather_agent.py)
  calls one MCP tool via `answer_with_tool`), or
- inherit the default `run()` tool loop, which iterates LLM ↔ tool calls up to
  `max_iterations` until the model returns a final answer.

`tool_names` semantics: `None` = load all permitted MCP tools, `[]` = no MCP
connection (pure-LLM agents like the Router/Planner/Synthesizer), `[...]` = load
only the named tools. Tools are also filtered by user role
([baseagent/permissions.py](baseagent/permissions.py)).

### Agents as services
Each agent ships a `META` ([`AgentMeta`](registry/agent_meta.py)) describing its
`agent_id`, version, capability tags, and input/output schemas. Running an agent
module calls [`run_agent(AgentClass, META)`](baseagent/agent_server.py), a harness
that:

- exposes the **A2A surface** — `POST /a2a` (JSON-RPC `message/send`) and
  `GET /.well-known/agent.json` (the Agent Card),
- **self-registers** its `AgentMeta` with the Registry on startup,
- **heartbeats** on an interval (and re-registers if the registry swept it),
- **deregisters** on shutdown.

This is why the orchestration graph imports no agent classes — discovery and
invocation are fully decoupled. Existing agents: `weather`, `outage`, `rag`.

---

## 5. Registry & A2A

**Registry Service** ([registry/service.py](registry/service.py), port 8002) is a
standalone FastAPI app and the single source of truth for "which agents are live."
Agents register/heartbeat; the Planner, Router, and Executor query it for the
capability menu. It replaced an older in-process static dict. Auth is an optional
bearer token (`REGISTRY_AUTH_TOKEN`). The orchestration app re-exposes the same
data at [`GET /registry`](main.py#L397) for the trace UI's live-discovery view.

**A2A** ([a2a/](a2a/)) is "A2A Hybrid" — formal agent-to-agent JSON-RPC messaging
layered on **centralized** registry discovery. [`A2AClient`](a2a/client.py)
resolves a target through the Registry, then sends `message/send` over HTTP. The
same client is used by the Executor (orchestration → agent) and by
[`BaseAgent.call_peer`](baseagent/base_agent.py#L173) (agent → agent), so agents
can fan work out to peers mid-run without importing each other.

---

## 6. Orchestration internals (the DAG)

- **Plan / Subtask / waves** — [planner/dag.py](planner/dag.py). A `Plan` is a list
  of `Subtask`s, each with `id`, `agent_id`, `args`, and `depends_on`. `Plan.waves()`
  computes execution waves via topological sort.
- **Orchestrator** splits the *decomposition* (compute waves, validate the DAG)
  from *execution* so each phase is independently observable in the trace.
- **Executor** runs each wave with `asyncio.gather`; the next wave starts only
  after the current one's tasks land on the blackboard. Args can reference
  upstream outputs with `${t1.text}` / `${t1.view.items.0.id}` syntax, resolved
  from the blackboard before the call.
- **Blackboard** ([orchestrator/blackboard.py](orchestrator/blackboard.py)) is the
  shared per-run workspace: a two-layer write — an in-memory dict mirrored onto
  `AgentState` (so downstream LangGraph nodes see it synchronously) **plus** Redis
  for cross-process visibility and audit (readable at
  [`GET /blackboard/{thread_id}/{run_id}`](main.py#L386)).

---

## 7. Memory subsystem

A layered design separating hot working memory from durable knowledge. Every
store **fails open** (disabled if its backend env var is unset or the client
package is missing) except the LangGraph checkpointer and guards.

| Layer | Backend | Module | Role |
|-------|---------|--------|------|
| **Graph checkpoint** | LangGraph saver | [memory/memory.py](memory/memory.py) | Per-thread graph state across turns; keyed by `thread_id`. |
| **Session / conversation** | MongoDB | [memory/mongo_store.py](memory/mongo_store.py) | Short-term recent-context cache (24h TTL) + durable `conversations` (no TTL, source of truth for listing/resuming). |
| **Blackboard** | Redis | [memory/redis_store.py](memory/redis_store.py) | Per-run shared working memory, 1h TTL. |
| **Long-term semantic** | Milvus | [memory/vector_store.py](memory/vector_store.py) | Embedding-based recall (`text-embedding-3-small`); the Planner recalls relevant prior context into its prompt. |
| **Durable facts** | MongoDB | [memory/facts_store.py](memory/facts_store.py) | Extracted key/value facts injected into planner/agent prompts; written by the Synthesizer. |
| **Commits / audit** | MongoDB | [memory/commit_store.py](memory/commit_store.py) | Append-only record of writes (shares the facts store's pymongo client). |

All durable stores have their indexes ensured at startup in
[`lifespan`](main.py#L34).

---

## 8. Security (guards)

[`LLMGuard`](security/llm_guard.py) wraps the local `llm-guard` library and is
**mandatory** — no enable/disable flag. Its models load eagerly at startup
([`get_llm_guard().warm()`](main.py#L75)) so a missing dependency or un-loadable
model **fails startup closed** rather than running the pipeline unprotected. Two
scanner classes:

- **Blocking** — prompt injection, toxicity/harmful, banned topics. A hit
  short-circuits the graph to a safe refusal.
- **Sanitizing** — PII (Anonymize/Sensitive) and credentials (Secrets). Never
  block; they redact in place, and the redaction is applied to the text that
  flows downstream regardless of the block decision.

`InputGuard`/`OutputGuard` ([security/guards.py](security/guards.py)) are the graph
nodes wrapping it. Optional INT8 ONNX quantized classifiers
([scripts/quantize_guard_models.py](scripts/quantize_guard_models.py)) speed up
inference; `TORCH_NUM_THREADS` caps per-model CPU oversubscription under
concurrency.

---

## 9. Observability

[`Observable`](observability/observable.py) auto-wraps each component's `run()` in
an MLflow span (span type per component: AGENT, CHAIN, etc.), so a full hierarchical
trace of every run is captured in MLflow
([observability/mlflow_setup.py](observability/mlflow_setup.py)). Structured JSON
logging ([observability/logging.py](observability/logging.py)) emits typed events
([baseagent/events.py](baseagent/events.py)) and can mirror spans to MLflow.

The [`POST /chat/trace`](main.py#L192) endpoint re-runs the same graph in
`stream_mode="updates"`, capturing each node's state delta as an animated step for
the explanation UI at `/trace.html` — including the live DB operations each node
performed.

---

## 10. HTTP surface (orchestration app)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | [`/chat`](main.py#L109) | Run the full pipeline; returns `{response, view}`. |
| `POST` | [`/chat/trace`](main.py#L192) | Same graph, step-by-step trace for the UI. |
| `GET` | [`/health`](main.py#L374) | Liveness. |
| `GET` | [`/state/{thread_id}`](main.py#L379) | Current LangGraph checkpoint for a thread. |
| `GET` | [`/blackboard/{thread_id}/{run_id}`](main.py#L386) | Redis-mirrored blackboard for one run. |
| `GET` | [`/registry`](main.py#L397) | Live discovered agents (proxies the Registry Service). |
| `GET` | [`/conversations`](main.py#L435) · `GET /conversations/{id}` · `DELETE /conversations/{id}` | List / resume / delete durable conversations. |
| `*` | `/` | Static frontend (`frontend/`, includes `index.html` and `trace.html`). |

---

## 11. Configuration

No-default required vars: `MCP_AUTH_TOKEN`, `OPENAI_BASE_URL`. Common optional
vars (see [.env.example](.env.example)): `OPENAI_MODEL` / `OPENAI_API_KEY` /
`OPENAI_TEMPERATURE`, per-component model overrides `ROUTER_MODEL` /
`PLANNER_MODEL`, `ROUTER_MIN_CONFIDENCE`, `MONGODB_URI`/`MONGODB_DB`, `REDIS_URL`,
Milvus connection vars, `MLFLOW_TRACKING_URI`/`MLFLOW_EXPERIMENT_NAME`,
`REGISTRY_AUTH_TOKEN`, `TORCH_NUM_THREADS`.

---

## 12. Running the system

The three tiers run as separate processes:

```bash
# 1. Registry Service (discovery source of truth)
python -m registry.service                 # :8002

# 2. Agent services — one process each (set AGENT_PORT per agent)
python -m agents.weather_agent             # self-registers with the Registry
python -m agents.outage_agent
python -m agents.rag_agent

# 3. Orchestration app + frontend
python main.py                             # :8000
```

Supporting infra (MongoDB, Redis, Milvus, an MLflow tracking server, and an
MCP server for tool-backed agents) is optional per-store — anything unset
degrades gracefully except the LLM, the LangGraph checkpointer, and the guards.

---

## Directory map

| Path | What lives there |
|------|------------------|
| [main.py](main.py) | FastAPI app, HTTP routes, request lifecycle |
| [state.py](state.py) | `AgentState` typed dict |
| [graph/](graph/) | LangGraph wiring + routing functions |
| [router/](router/) | Fast intent triage + local intent classifier |
| [planner/](planner/) | DAG plan generation, `Plan`/`Subtask`, JSON parsing |
| [orchestrator/](orchestrator/) | Wave decomposition, Executor, Blackboard |
| [gate/](gate/) | Completion gate / re-plan decision |
| [synthesizer/](synthesizer/) | Final-answer composition + fact extraction |
| [security/](security/) | Mandatory llm-guard input/output scanning |
| [baseagent/](baseagent/) | `BaseAgent`, LLM/MCP clients, agent service harness |
| [agents/](agents/) | Concrete agent services (weather, outage, rag) |
| [registry/](registry/) | Registry Service, client, `AgentMeta`, store |
| [a2a/](a2a/) | A2A JSON-RPC client/types + Agent Card |
| [memory/](memory/) | Mongo / Redis / Milvus / facts / commit stores + checkpointer |
| [observability/](observability/) | MLflow setup, `Observable`, JSON logging |
| [mcp_server/](mcp_server/), [mcpconfig/](mcpconfig/) | MCP tool server + config |
| [frontend/](frontend/) | Chat UI (`index.html`) and trace UI (`trace.html`) |
