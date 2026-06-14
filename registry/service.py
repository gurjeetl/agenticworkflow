"""Standalone Registry/Discovery Service.

Independent FastAPI app that agents self-register with (and heartbeat to), and
that the Planner/Executor query for agent discovery. Replaces the old in-process
static dict as the registry's source of truth.

Run: python -m registry.service
Endpoint: http://127.0.0.1:8002
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException

from observability import configure_logging, get_logger
from registry.contracts import (
    DeregisterRequest,
    HeartbeatRequest,
    HeartbeatResponse,
    ListResponse,
    RegisterRequest,
    RegisterResponse,
)
from registry.store import get_registry_store

load_dotenv()
configure_logging()
_log = get_logger(__name__)


def require_auth(authorization: str | None = Header(None)) -> None:
    """Bearer-token gate. No-op when REGISTRY_AUTH_TOKEN is unset (local dev)."""
    token = os.getenv("REGISTRY_AUTH_TOKEN")
    if not token:
        return
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="invalid registry token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_registry_store()
    await store.ensure_indexes()
    _log.info("registry.indexes_ensured", extra={"attrs": {"ttl_seconds": store.ttl_seconds}})
    yield
    store.close()


app = FastAPI(title="Agent Registry Service", lifespan=lifespan)


def _heartbeat_interval() -> int:
    return int(os.getenv("REGISTRY_HEARTBEAT_SECONDS", str(max(1, get_registry_store().ttl_seconds // 3))))


@app.post("/register", response_model=RegisterResponse, dependencies=[Depends(require_auth)])
async def register(req: RegisterRequest) -> RegisterResponse:
    store = get_registry_store()
    meta = req.meta
    if not meta.endpoint:
        raise HTTPException(status_code=422, detail="meta.endpoint is required for remote agents")
    if not meta.instance_id:
        meta.instance_id = uuid.uuid4().hex
    await store.upsert(meta)
    _log.info(
        "registry.register",
        extra={"attrs": {"agent_id": meta.agent_id, "instance_id": meta.instance_id, "endpoint": meta.endpoint}},
    )
    return RegisterResponse(
        instance_id=meta.instance_id,
        ttl_seconds=store.ttl_seconds,
        heartbeat_interval_seconds=_heartbeat_interval(),
    )


@app.post("/heartbeat", response_model=HeartbeatResponse, dependencies=[Depends(require_auth)])
async def heartbeat(req: HeartbeatRequest) -> HeartbeatResponse:
    store = get_registry_store()
    known = await store.heartbeat(req.instance_id, req.status)
    if not known:
        _log.warning("registry.heartbeat_unknown", extra={"attrs": {"instance_id": req.instance_id}})
    return HeartbeatResponse(ok=known, known=known)


@app.post("/deregister", dependencies=[Depends(require_auth)])
async def deregister(req: DeregisterRequest) -> dict:
    store = get_registry_store()
    removed = await store.deregister(req.instance_id)
    _log.info("registry.deregister", extra={"attrs": {"instance_id": req.instance_id, "removed": removed}})
    return {"ok": True, "removed": removed}


@app.get("/agents", response_model=ListResponse, dependencies=[Depends(require_auth)])
async def list_agents(agent_id: str | None = None, tag: str | None = None) -> ListResponse:
    store = get_registry_store()
    agents = await (store.get_agent(agent_id) if agent_id else store.list_active())
    if tag:
        agents = [m for m in agents if tag in (m.capability_tags or [])]
    return ListResponse(agents=agents)


@app.get("/agents/{agent_id}", response_model=ListResponse, dependencies=[Depends(require_auth)])
async def get_agent(agent_id: str) -> ListResponse:
    store = get_registry_store()
    return ListResponse(agents=await store.get_agent(agent_id))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("REGISTRY_PORT", "8002"))
    uvicorn.run("registry.service:app", host="127.0.0.1", port=port)
