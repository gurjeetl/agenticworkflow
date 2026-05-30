# Future: A2A Protocol Support

Plan for making this project's agents speak Google's **Agent2Agent (A2A)** protocol so they are independently addressable, discoverable, and composable with agents built by other teams.

---

## Guiding principles

- **A2A is transport, not implementation.** [BaseAgent](baseagent/base_agent.py) stays the substrate. The A2A layer is a thin wrapper that exposes any `BaseAgent` subclass over JSON-RPC.
- **Subclasses don't change.** Adding A2A must not require edits to [HotelAgent](agents/hotel_agent.py) / [WeatherAgent](agents/weather_agent.py) beyond declaring a `skills` list.
- **Symmetry.** BaseAgent is both an A2A *server* (other agents call it) and an A2A *client* (it calls other agents) — the same way MCP gives it tool-client symmetry today.
- **Migration is phased.** In-process LangGraph orchestration keeps working at every step. We only flip to network calls when each piece is ready.

---

## Target architecture (end state)

```
                    ┌────────────────────────┐
   external caller →│  router-agent (A2A)    │
                    │  /.well-known/agent.json│
                    └──────────┬─────────────┘
                               │ A2A JSON-RPC
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       weather-agent     hotel-agent      <other team>
       (A2A server)      (A2A server)     (A2A server)
              │                │
              ▼                ▼
           MCP tools        MCP tools
```

- Each agent: own FastAPI app, own container, own Agent Card, own auth.
- Router holds A2A client stubs resolved from a registry (or static config).
- Tasks persist in MongoDB; long work runs on a worker; SSE streams progress.

---

## Phased migration

### Phase 0 — Foundations on `BaseAgent` (in-process, no network changes)

Goal: get the abstractions right before any service split.

- [ ] Add `skills: list[AgentSkill]` class attribute on `BaseAgent` (id, name, description, examples, input/output modes).
- [ ] Add `agent_card() -> dict` on `BaseAgent` that derives the Agent Card from `system_prompt`, `tool_names`, `skills`, plus env-driven `url` / `version`.
- [ ] Add `to_a2a_task(state: AgentState) -> Task` and `from_a2a_message(msg) -> AgentState` translators on `BaseAgent`. Subclasses keep returning plain `AgentState`.
- [ ] Add `BaseAgent.a2a_call(agent_url, message, *, stream=False)` outbound client — mirrors `call_mcp_tool` but for agent-to-agent.
- [ ] Unit tests: every existing subclass produces a valid Agent Card and round-trips a task.

**Exit criteria:** no behavior change; `pytest` green; cards validate against A2A JSON schema.

### Phase 1 — Single composite A2A endpoint (option 5b from design discussion)

Goal: external clients can reach the *whole system* over A2A without splitting services yet.

- [ ] New `POST /a2a` JSON-RPC endpoint on [main.py](main.py) alongside existing `/chat`. Implements `message/send`, `tasks/get`, `tasks/cancel`.
- [ ] New `GET /.well-known/agent.json` aggregating skills from all registered agents.
- [ ] Task store backed by [mongo_store.py](memory/mongo_store.py): `tasks` collection keyed by `taskId`, with `status`, `history`, `artifacts`, `contextId`.
- [ ] `message/send` still calls `graph.invoke()` synchronously — task transitions `submitted → working → completed` in one shot.
- [ ] Bearer auth via existing `MCP_AUTH_TOKEN` style, declared in the Card's `securitySchemes`.

**Exit criteria:** the [agent_diagram.html](agent_diagram.html) frontend keeps working via `/chat`; an A2A client (e.g. `a2a-cli`) can hit `/a2a` and round-trip a task.

### Phase 2 — Async execution + streaming

Goal: long-running tasks, progress events, and `tasks/get` polling.

- [ ] Worker process (start with [Arq](https://arq-docs.helpmanual.io/) on Redis; revisit Temporal at Phase 4). `message/send` enqueues, returns `submitted` immediately.
- [ ] Implement `message/stream` (SSE) — fed by an event bus that [Observable](observability/observable.py) already produces. Per-event types: `status-update`, `artifact-update`, `final`.
- [ ] `tasks/cancel` flips a flag the agent loop checks between iterations in [base_agent.py:205](baseagent/base_agent.py#L205).
- [ ] Idempotency keys on inbound `message/send`.

**Exit criteria:** a client can `message/stream` and see tool-call events live; cancellation works mid-loop.

### Phase 3 — Per-agent services (option 5a)

Goal: each agent independently deployable.

- [ ] Extract `weather-agent`, `hotel-agent`, `router-agent` into their own FastAPI entrypoints (still in this monorepo, separate processes).
- [ ] Router becomes an A2A *client* — resolves skill → URL from `AGENT_REGISTRY` env (JSON map) or a small registry service.
- [ ] [graph_builder.py](graph/graph_builder.py) keeps the LangGraph topology, but the leaf nodes call `BaseAgent.a2a_call()` instead of invoking subclasses directly.
- [ ] Dockerfile per agent; `docker-compose.yml` for local dev.
- [ ] mTLS or OIDC client-credentials between services. Drop static bearer.

**Exit criteria:** kill the weather process → only weather requests fail; others keep working. Adding a 4th agent from another team is a config change, not a code change.

### Phase 4 — Enterprise hardening

Defer until there are >3 agents or external consumers.

- [ ] Replace Arq with **Temporal** for retries, timeouts, compensation across multi-agent calls.
- [ ] Agent Card schema registry; CI validates cards on every PR; semver enforced on breaking skill changes.
- [ ] Service mesh (Envoy/Istio) terminating mTLS, rate limits, per-caller quotas.
- [ ] Distributed tracing: propagate `traceparent` through every A2A hop so [MLflow](observability/mlflow_setup.py) spans stitch across services.
- [ ] Per-skill SLO dashboards; audit log of every inbound A2A request (caller, skill, taskId).
- [ ] Push notifications (`pushNotificationConfig`) so callers can register webhooks instead of holding SSE connections.

---

## Open questions

- **Registry choice.** Static JSON env → small in-house registry → Consul? Decide at Phase 3.
- **Worker choice.** Arq is cheapest; Temporal is the enterprise answer. Don't pre-pay Temporal complexity.
- **Identity.** SPIFFE/SPIRE vs. existing IdP. Tied to whatever the broader org standardizes on.
- **Multi-tenant context.** `contextId` semantics — is it our existing `thread_id`, or scoped per tenant? Likely `tenant:thread_id`.
- **Composite vs per-agent card** in Phase 1. Composite is simpler but locks consumers into "one big agent" mental model. Could publish both.

---

## Non-goals

- Replacing MCP. MCP stays the tool transport; A2A is the agent transport.
- Replacing LangGraph. The orchestration topology is orthogonal to how nodes talk to each other.
- Building a UI for agent discovery — use existing A2A tooling (`a2a-cli`, `a2a-inspector`).

---

## Reference

- A2A spec: https://google.github.io/A2A/
- Agent Card JSON schema: https://google.github.io/A2A/specification/agent-card/
- Existing in-repo design discussion: see chat transcript that produced this file.
