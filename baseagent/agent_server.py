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
        from baseagent.agent_server import run_agent
        run_agent(WeatherAgent, META)
"""
from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException

from a2a.agent_card import to_agent_card
from a2a.types import (
    ERR_AGENT_EXECUTION,
    ERR_INVALID_PARAMS,
    ERR_METHOD_NOT_FOUND,
    METHOD_MESSAGE_SEND,
    JsonRpcError,
    JsonRpcResponse,
    Message,
    data_part,
    get_data,
    text_part,
)
from observability import configure_logging, get_logger
from orchestrator.task_state import build_task_state
from registry.agent_meta import AgentMeta

load_dotenv()
configure_logging()
_log = get_logger(__name__)


# --- Config helpers ---------------------------------------------------------
def _advertised_endpoint() -> str:
    host = os.getenv("AGENT_ADVERTISE_HOST") or os.getenv("AGENT_HOST", "127.0.0.1")
    port = os.getenv("AGENT_ADVERTISE_PORT") or os.getenv("AGENT_PORT", "8010")
    return f"http://{host}:{port}"


def _registry_base() -> str:
    return os.getenv("REGISTRY_URL", "http://127.0.0.1:8002").rstrip("/")


def _registry_headers() -> dict:
    token = os.getenv("REGISTRY_AUTH_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def require_a2a_auth(authorization: str | None = Header(None)) -> None:
    token = os.getenv("AGENT_INVOKE_TOKEN")
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


# --- App factory ------------------------------------------------------------
def create_agent_app(agent_cls: type, meta: AgentMeta) -> FastAPI:
    agent = agent_cls()  # loads LLM + MCP from env, once
    meta = meta.model_copy(update={"endpoint": _advertised_endpoint(), "instance_id": uuid.uuid4().hex})
    default_interval = int(os.getenv("REGISTRY_HEARTBEAT_SECONDS", "30"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = httpx.AsyncClient(timeout=float(os.getenv("REGISTRY_TIMEOUT_S", "3")))
        interval = await _register(client, meta) or default_interval
        hb_task = asyncio.create_task(_heartbeat_loop(client, meta, interval))
        try:
            yield
        finally:
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
        return {"status": "ok", "agent_id": meta.agent_id, "instance_id": meta.instance_id}

    @app.get("/.well-known/agent.json")
    async def agent_card() -> dict:
        """Formal A2A discovery document. Routing goes through the Registry, but
        the card is served here too for A2A interoperability."""
        return to_agent_card(meta).model_dump(mode="json")

    @app.post("/a2a", dependencies=[Depends(require_a2a_auth)])
    async def a2a(body: dict) -> dict:
        """A2A JSON-RPC 2.0 endpoint. Handles ``message/send``.

        Args travel in a request DataPart (``{"args": {...}}``); invocation
        context (task_id, run_id, thread_id, blackboard) travels in the message
        ``metadata``. The reply is an agent-role Message: a TextPart with the
        agent's answer and an optional DataPart carrying a structured ``view``.
        """
        rpc_id = body.get("id")

        def _err(code: int, message: str) -> dict:
            return JsonRpcResponse(id=rpc_id, error=JsonRpcError(code=code, message=message)).model_dump(mode="json")

        if body.get("method") != METHOD_MESSAGE_SEND:
            return _err(ERR_METHOD_NOT_FOUND, f"unsupported method '{body.get('method')}'")
        try:
            in_msg = Message.model_validate((body.get("params") or {}).get("message") or {})
        except Exception as e:  # malformed message payload
            return _err(ERR_INVALID_PARAMS, f"invalid message: {e}")

        meta_in = in_msg.metadata or {}
        args = (get_data(in_msg) or {}).get("args") or {}
        state = build_task_state(
            task_id=meta_in.get("task_id") or in_msg.taskId or "",
            agent_id=meta_in.get("agent_id") or meta.agent_id,
            args=args,
            thread_id=meta_in.get("thread_id") or in_msg.contextId or "",
            run_id=meta_in.get("run_id") or "",
            blackboard=meta_in.get("blackboard") or {},
        )
        result_state = await asyncio.to_thread(agent.run, state)
        if result_state.get("error"):
            return _err(ERR_AGENT_EXECUTION, str(result_state["error"]))

        parts = [text_part(result_state.get("final_output") or "")]
        view = result_state.get("view")
        if view:
            parts.append(data_part({"view": view}))
        out_msg = Message(
            role="agent",
            messageId=uuid.uuid4().hex,
            taskId=meta_in.get("task_id"),
            contextId=meta_in.get("thread_id"),
            parts=parts,
            metadata={"agent_id": meta.agent_id},
        )
        return JsonRpcResponse(id=rpc_id, result=out_msg.model_dump(mode="json")).model_dump(mode="json")

    return app


def run_agent(agent_cls: type, meta: AgentMeta) -> None:
    host = os.getenv("AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("AGENT_PORT", "8010"))
    uvicorn.run(create_agent_app(agent_cls, meta), host=host, port=port)
