"""Reusable harness that turns a BaseAgent subclass into a remote service.

Each agent now runs as its own process. This harness gives every agent the same
behavior without per-agent FastAPI boilerplate:

  * exposes the formal A2A surface — ``POST /a2a`` (JSON-RPC ``message/send``,
    the contract the Executor and peer agents call) and ``GET
    /.well-known/agent.json`` (the Agent Card),
  * self-registers its :class:`AgentMeta` with the Registry Service on startup,
  * heartbeats on an interval so the registry keeps it "live" (TTL),
  * re-registers automatically if the registry swept/restarted,
  * deregisters on shutdown.

The agent's own LLM + MCP wiring is unchanged — it loads from the same env as
before (OPENAI_*, MCP_SERVER_URL, ...).

Usage (per agent module)::

    if __name__ == "__main__":
        from genie.agents.server import run_agent
        run_agent(WeatherAgent, META)
"""
from __future__ import annotations

import asyncio
import socket
import uuid
from contextlib import asynccontextmanager

import json
import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from genie.a2a.agent_card import to_agent_card
from genie.a2a.types import (
    METHOD_MESSAGE_SEND,
    METHOD_MESSAGE_STREAM,
    METHOD_TASKS_CANCEL,
    METHOD_TASKS_GET,
    Artifact,
    InvalidParamsError,
    JSONRPCError,
    JSONRPCErrorResponse,
    Message,
    MethodNotFoundError,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskNotCancelableError,
    TaskNotFoundError,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    UnsupportedOperationError,
    data_part,
    get_data,
    text_part,
)
from genie.messaging import Dedup, get_broker
from genie.messaging.broker import Broker, BusMessage
from genie.messaging.envelope import (
    HDR_ATTEMPT,
    HDR_CORRELATION_ID,
    HDR_ERROR,
    HDR_FROM,
    HDR_KIND,
    HDR_REPLY_TO,
    HDR_THREAD_ID,
    KIND_DEAD_LETTER,
    KIND_REPLY,
    dlq_topic,
    inbox_topic,
    reply_topic,
)
from genie.observability import configure_logging, get_logger
from genie.platform.config import get_settings
from genie.platform.redis import redis_enabled
from genie.agents.task_state import build_task_state
from genie.agents.task_store import TaskStore
from genie.application.state import AgentState
from genie.registry.agent_meta import AgentMeta

load_dotenv()
configure_logging()
_log = get_logger(__name__)


# --- Config helpers ---------------------------------------------------------
def _advertised_endpoint(port: int) -> str:
    """Base URL peers should use to reach this agent at the given bound ``port``.

    ``agent_advertise_host`` / ``agent_advertise_port`` override the bind values
    for NAT / container scenarios where peers reach the agent at a different
    address than it binds locally.
    """
    _s = get_settings()
    host = _s.agent_advertise_host or _s.agent_host
    adv_port = _s.agent_advertise_port or port
    return f"http://{host}:{adv_port}"


def _registry_base() -> str:
    """Base URL of the Registry Service this agent registers/heartbeats against."""
    return get_settings().registry_url.rstrip("/")


def _registry_headers() -> dict:
    """Auth headers for registry calls, empty when no registry auth token is set."""
    token = get_settings().registry_auth_token
    return {"Authorization": f"Bearer {token}"} if token else {}


def require_a2a_auth(authorization: str | None = Header(None)) -> None:
    """FastAPI dependency guarding ``/a2a``; a no-op unless an A2A invoke token is set."""
    token = get_settings().agent_invoke_token
    if not token:
        return
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="invalid A2A token")


# --- Registry interactions --------------------------------------------------
async def _register(client: httpx.AsyncClient, meta: AgentMeta) -> int | None:
    """Register; return the heartbeat interval (seconds) or None on failure."""
    try:
        resp = await client.post(
            f"{_registry_base()}/register",
            json={"meta": meta.model_dump(mode="json")},
            headers=_registry_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        _log.info(
            "agent.registered",
            extra={"attrs": {"agent_id": meta.agent_id, "instance_id": meta.instance_id}},
        )
        return int(data.get("heartbeat_interval_seconds") or 0) or None
    except Exception as e:
        _log.warning("agent.register_failed", extra={"attrs": {"error": str(e)}})
        return None


async def _heartbeat_loop(client: httpx.AsyncClient, meta: AgentMeta, interval: int) -> None:
    """Periodically heartbeat the registry, re-registering if it forgot or restarted us."""
    while True:
        await asyncio.sleep(interval)
        try:
            resp = await client.post(
                f"{_registry_base()}/heartbeat",
                json={"instance_id": meta.instance_id, "status": meta.status},
                headers=_registry_headers(),
            )
            resp.raise_for_status()
            if not resp.json().get("known", False):
                _log.info("agent.reregister_unknown", extra={"attrs": {"agent_id": meta.agent_id}})
                await _register(client, meta)
        except Exception as e:
            _log.warning("agent.heartbeat_failed", extra={"attrs": {"error": str(e)}})
            await _register(client, meta)  # self-heal: registry restarted/unreachable


# --- Task execution helpers -------------------------------------------------
def _agent_message(meta: AgentMeta, task_id: str, context_id: str, result_state: AgentState) -> Message:
    """Build the agent-role reply Message from a finished AgentState.

    A text part carries ``final_output``; an optional data part carries a
    structured ``view`` — the same shape ``message/send`` has always returned.
    """
    parts = [text_part(result_state.get("final_output") or "")]
    view = result_state.get("view")
    if view:
        parts.append(data_part({"view": view}))
    return Message(
        role=Role.agent,
        message_id=uuid.uuid4().hex,
        task_id=task_id,
        context_id=context_id,
        parts=parts,
        metadata={"agent_id": meta.agent_id},
    )


def _completed_task(meta: AgentMeta, task_id: str, context_id: str, result_state: AgentState) -> Task:
    """Wrap a finished AgentState into a terminal A2A :class:`Task`.

    ``error`` on the state → a ``failed`` task carrying the error text; otherwise
    a ``completed`` task whose ``status.message`` is the agent's reply, with the
    structured ``view`` (when present) also surfaced as an artifact.
    """
    error = result_state.get("error")
    if error:
        fail_msg = Message(
            role=Role.agent,
            message_id=uuid.uuid4().hex,
            task_id=task_id,
            context_id=context_id,
            parts=[text_part(str(error))],
            metadata={"agent_id": meta.agent_id},
        )
        return Task(id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.failed, message=fail_msg))

    msg = _agent_message(meta, task_id, context_id, result_state)
    artifacts = None
    view = result_state.get("view")
    if view:
        artifacts = [Artifact(artifact_id=uuid.uuid4().hex, name="view", parts=[data_part({"view": view})])]
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=TaskState.completed, message=msg),
        artifacts=artifacts,
    )


def _resolve_ids(in_msg: Message) -> tuple[str, str]:
    """Resolve the ``(task_id, context_id)`` for an inbound message.

    Both are required, non-null strings on the A2A ``Task`` and streaming events,
    so when the caller supplies neither (e.g. the A2A Inspector, which sends no
    thread/context), the server mints them — matching the spec's "server assigns
    a contextId for a new interaction" rule.
    """
    meta_in = in_msg.metadata or {}
    task_id = meta_in.get("task_id") or in_msg.task_id or uuid.uuid4().hex
    context_id = meta_in.get("thread_id") or in_msg.context_id or uuid.uuid4().hex
    return task_id, context_id


async def _run_task(agent, meta: AgentMeta, in_msg: Message, task_id: str, context_id: str) -> Task:
    """Run the agent for one inbound Message and return its terminal Task.

    Rebuilds the same narrow ``AgentState`` the pre-1.2 path used (via
    ``build_task_state``) and runs the synchronous ``agent.run`` on a worker
    thread so the event loop stays free to service concurrent ``tasks/get`` and
    streaming requests. ``task_id``/``context_id`` are resolved once by the
    caller (:func:`_resolve_ids`) so a stream's events and its terminal task all
    carry the same ids.
    """
    meta_in = in_msg.metadata or {}
    args = (get_data(in_msg) or {}).get("args") or {}
    state = build_task_state(
        task_id=task_id,
        agent_id=meta_in.get("agent_id") or meta.agent_id,
        args=args,
        thread_id=context_id,
        run_id=meta_in.get("run_id") or "",
        blackboard=meta_in.get("blackboard") or {},
    )
    result_state = await asyncio.to_thread(agent.run, state)
    return _completed_task(meta, task_id, context_id, result_state)


# --- Async transport: Kafka inbox consumer -----------------------------------
async def _handle_inbox_message(
    broker: Broker, agent, meta: AgentMeta, tasks: TaskStore, dedup: Dedup, bm: BusMessage
) -> None:
    """Process one bus-delivered request: dedup → run → reply (poison → DLQ).

    Mirrors the diagram's consumer contract exactly:

    * A payload that fails schema validation is a **poison pill** — every retry
      would fail identically, so it goes straight to the DLQ with the parse
      error attached, and **no reply** is produced (the caller's deadline sweep
      or the Supervisor unblocks the waiting run).
    * A redelivered ``(correlation_id, attempt)`` is dropped by the dedup claim,
      so at-least-once delivery never double-runs the agent.
    * A *business* failure (the agent ran and returned an error) is NOT dead-
      lettered — it comes back as a ``failed`` Task reply, exactly like the
      synchronous path, so the Executor/Gate handle it uniformly.
    """
    headers = bm.headers
    cid = headers.get(HDR_CORRELATION_ID, "")
    attempt = headers.get(HDR_ATTEMPT, "1")
    try:
        in_msg = Message.model_validate(json.loads(bm.value))
    except Exception as e:  # poison pill: no retry can ever succeed
        _log.warning("a2a.bus.poison_pill", extra={"attrs": {"cid": cid, "error": str(e)}})
        await broker.produce(
            dlq_topic(),
            value=bm.value,
            key=bm.key,
            headers={
                **headers,
                HDR_KIND: KIND_DEAD_LETTER,
                HDR_FROM: meta.agent_id,
                HDR_ERROR: f"schema_validation_failed: {e}",
            },
        )
        return

    if cid and not await dedup.claim_inbox(meta.agent_id, cid, attempt):
        _log.info("a2a.bus.duplicate_dropped", extra={"attrs": {"cid": cid, "attempt": attempt}})
        return

    task_id, context_id = _resolve_ids(in_msg)
    task = await _run_task(agent, meta, in_msg, task_id, context_id)
    tasks.put(task)
    await broker.produce(
        headers.get(HDR_REPLY_TO) or reply_topic(),
        value=task.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8"),
        key=headers.get(HDR_THREAD_ID) or context_id,
        headers={
            HDR_KIND: KIND_REPLY,
            HDR_CORRELATION_ID: cid,
            HDR_ATTEMPT: attempt,
            HDR_FROM: meta.agent_id,
            HDR_THREAD_ID: headers.get(HDR_THREAD_ID, context_id),
        },
    )
    _log.info(
        "a2a.bus.replied",
        extra={"attrs": {"cid": cid, "state": task.status.state.value, "task_id": task.id}},
    )


async def _consume_inbox(broker: Broker, agent, meta: AgentMeta, tasks: TaskStore) -> None:
    """Consume this agent's inbox topic forever; one handler call per record.

    Handler errors are logged and the loop continues — one bad record must never
    kill the consumer (the record was auto-committed; dedup + the caller's
    deadline path own recovery).
    """
    topic = meta.inbox_topic or inbox_topic(meta.agent_id)
    group = f"{get_settings().bus_topic_prefix}-agent-{meta.agent_id}"
    dedup = Dedup()
    _log.info("a2a.bus.consuming", extra={"attrs": {"topic": topic, "group": group}})
    async for bm in broker.consume([topic], group=group):
        try:
            await _handle_inbox_message(broker, agent, meta, tasks, dedup, bm)
        except Exception as e:
            _log.error("a2a.bus.handler_failed", extra={"attrs": {"error": str(e)}}, exc_info=True)


def _rpc_result(rpc_id, payload) -> dict:
    """Build a JSON-RPC success response with an a2a-sdk payload as its ``result``.

    ``by_alias`` emits the SDK's camelCase wire names; ``exclude_none`` drops null
    optional fields so strict A2A SDK clients (e.g. A2A Inspector) validate the
    frame cleanly.
    """
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "result": payload.model_dump(mode="json", by_alias=True, exclude_none=True),
    }


def _rpc_error(rpc_id, error: JSONRPCError) -> dict:
    """Build a JSON-RPC error response from an a2a-sdk error object (standard code)."""
    return JSONRPCErrorResponse(id=rpc_id, error=error).model_dump(mode="json", by_alias=True, exclude_none=True)


def _sse(rpc_id, payload) -> str:
    """Frame one JSON-RPC result as a Server-Sent Events ``data:`` block."""
    return f"data: {json.dumps(_rpc_result(rpc_id, payload))}\n\n"


# --- App factory ------------------------------------------------------------
def create_agent_app(agent_cls: type, meta: AgentMeta, port: int) -> FastAPI:
    """Build the FastAPI app exposing one agent as an A2A service.

    Instantiates the agent once, stamps the advertised endpoint (derived from the
    resolved bound ``port``) and a fresh instance id onto its meta, and wires the
    registry register/heartbeat/deregister lifecycle plus the health, agent-card,
    and ``/a2a`` routes.
    """
    agent = agent_cls()  # loads LLM + MCP from env, once
    meta = meta.model_copy(update={"endpoint": _advertised_endpoint(port), "instance_id": uuid.uuid4().hex})
    default_interval = get_settings().registry_heartbeat_seconds
    tasks = TaskStore()  # in-process Task store backing tasks/get + tasks/cancel

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Register + start heartbeats (+ the Kafka inbox consumer in async mode)."""
        client = httpx.AsyncClient(timeout=get_settings().registry_timeout_s)
        interval = await _register(client, meta) or default_interval
        hb_task = asyncio.create_task(_heartbeat_loop(client, meta, interval))

        # Async transport: consume this agent's inbox topic when opted in.
        bus_task: asyncio.Task | None = None
        broker = None
        settings = get_settings()
        if settings.kafka_enabled and meta.transport in ("kafka", "both"):
            if not redis_enabled():
                raise RuntimeError(
                    "kafka_enabled=True requires Redis (redis_url) for bus dedup — "
                    "configure REDIS_URL or disable the async transport"
                )
            broker = get_broker()
            bus_task = asyncio.create_task(_consume_inbox(broker, agent, meta, tasks))
        try:
            yield
        finally:
            if bus_task is not None:
                bus_task.cancel()
            if broker is not None:
                await broker.close()
            hb_task.cancel()
            try:
                await client.post(
                    f"{_registry_base()}/deregister",
                    json={"instance_id": meta.instance_id},
                    headers=_registry_headers(),
                )
            except Exception:
                pass
            await client.aclose()

    app = FastAPI(title=f"agent:{meta.agent_id}", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        """Liveness probe identifying this agent and its current instance."""
        return {"status": "ok", "agent_id": meta.agent_id, "instance_id": meta.instance_id}

    # Both well-known paths: `agent.json` (older A2A) and `agent-card.json` (A2A
    # v0.3+, used by tools like the A2A Inspector). Serving both keeps every client
    # version interoperable.
    @app.get("/.well-known/agent.json")
    @app.get("/.well-known/agent-card.json")
    async def agent_card() -> dict:
        """Formal A2A discovery document. Routing goes through the Registry, but
        the card is served here too for A2A interoperability."""
        return to_agent_card(meta).model_dump(mode="json", by_alias=True, exclude_none=True)

    def _incoming(body: dict) -> Message:
        """Parse the inbound A2A Message from a JSON-RPC request body (raises on bad input)."""
        return Message.model_validate((body.get("params") or {}).get("message") or {})

    async def _stream_events(rpc_id, in_msg: Message):
        """Yield the A2A SSE frames for one ``message/stream`` invocation.

        Emits ``submitted`` → ``working`` → run the agent → an artifact frame (when
        the reply carries content) → a terminal ``completed``/``failed`` status
        (``final=true``). Agents run synchronously, so these are lifecycle events,
        not token-level deltas. The terminal Task is stored for later ``tasks/get``.
        """
        task_id, context_id = _resolve_ids(in_msg)
        submitted = Task(id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.submitted))
        yield _sse(rpc_id, submitted)
        yield _sse(rpc_id, TaskStatusUpdateEvent(task_id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.working), final=False))

        task = await _run_task(agent, meta, in_msg, task_id, context_id)
        tasks.put(task)
        if task.artifacts:
            for art in task.artifacts:
                yield _sse(rpc_id, TaskArtifactUpdateEvent(task_id=task.id, context_id=task.context_id, artifact=art, last_chunk=True))
        yield _sse(rpc_id, TaskStatusUpdateEvent(task_id=task.id, context_id=task.context_id, status=task.status, final=True))

    @app.post("/a2a", dependencies=[Depends(require_a2a_auth)])
    async def a2a(body: dict):
        """A2A JSON-RPC 2.0 endpoint (protocol 1.0.0 / JSON binding).

        Handles ``message/send`` (returns a terminal :class:`Task`),
        ``message/stream`` (SSE stream of task lifecycle events), ``tasks/get``
        and ``tasks/cancel``. Args travel in a request DataPart
        (``{"args": {...}}``); invocation context (task_id, run_id, thread_id,
        blackboard) travels in the message ``metadata``. Errors use the standard
        a2a-sdk error classes, so JSON-RPC codes are spec-correct.
        """
        rpc_id = body.get("id")
        method = body.get("method")

        if method == METHOD_MESSAGE_SEND:
            try:
                in_msg = _incoming(body)
            except Exception as e:
                return _rpc_error(rpc_id, InvalidParamsError(data={"detail": f"invalid message: {e}"}))
            task = await _run_task(agent, meta, in_msg, *_resolve_ids(in_msg))
            tasks.put(task)
            return _rpc_result(rpc_id, task)

        if method == METHOD_MESSAGE_STREAM:
            if not meta.supports_streaming:  # keep the endpoint honest vs. the card
                return _rpc_error(rpc_id, UnsupportedOperationError())
            try:
                in_msg = _incoming(body)
            except Exception as e:
                return _rpc_error(rpc_id, InvalidParamsError(data={"detail": f"invalid message: {e}"}))
            return StreamingResponse(_stream_events(rpc_id, in_msg), media_type="text/event-stream")

        if method == METHOD_TASKS_GET:
            task = _lookup_task(body)
            return _rpc_result(rpc_id, task) if task else _rpc_error(rpc_id, TaskNotFoundError())

        if method == METHOD_TASKS_CANCEL:
            task = _lookup_task(body)
            if task is None:
                return _rpc_error(rpc_id, TaskNotFoundError())
            # Runs are synchronous, so a stored task is always terminal — the spec
            # requires TaskNotCancelableError for that case.
            return _rpc_error(rpc_id, TaskNotCancelableError())

        return _rpc_error(rpc_id, MethodNotFoundError())

    def _lookup_task(body: dict) -> Task | None:
        """Fetch the stored Task referenced by ``params.id`` (None if absent)."""
        task_id = (body.get("params") or {}).get("id")
        return tasks.get(task_id) if task_id else None

    return app


def run_agent(agent_cls: type, meta: AgentMeta, port: int | None = None) -> None:
    """Module ``__main__`` entry point: serve the agent app with uvicorn.

    Port resolution (highest priority first):

    1. ``AGENT_PORT`` env var / ``agent_port`` in YAML (a deploy/container pin) —
       applies in any environment.
    2. The per-agent ``port`` passed here — a stable, memorable default for manual
       testing (e.g. 8010). A developer affordance, not a contract: it applies
       ONLY in development mode (``ENVIRONMENT=development``, the default), so it
       never silently pins a port in production.
    3. ``0`` → an OS-assigned ephemeral port. The real bound port is advertised to
       the registry, so discovery (by ``agent_id``) keeps working with no per-agent
       port config — the path that scales to many agents and the prod default.
    """
    _s = get_settings()
    if _s.agent_port is not None:
        resolved = _s.agent_port                       # explicit pin (env/YAML), any mode
    elif _s.is_development and port is not None:
        resolved = port                                # dev-only stable default
    else:
        resolved = 0                                   # ephemeral + discovery (prod/unset)
    host = _s.agent_host

    if resolved != 0:
        uvicorn.run(create_agent_app(agent_cls, meta, resolved), host=host, port=resolved)
        return

    # Ephemeral: bind first so we can advertise the port the OS actually assigned.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, 0))
    bound_port = sock.getsockname()[1]
    app = create_agent_app(agent_cls, meta, bound_port)
    uvicorn.Server(uvicorn.Config(app)).run(sockets=[sock])
