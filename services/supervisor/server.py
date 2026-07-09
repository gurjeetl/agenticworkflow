"""Standalone A2A Supervisor Service (Phase 2 control plane).

Independent process that owns the async-A2A failure ladder (extend → retry →
dead-letter) and permanently consumes ``genie.dlq``, unblocking waiting runs
with ``step.cancelled`` control messages. See ``genie.messaging.supervisor``
for the logic; this is the thin FastAPI runner (health endpoint + lifespan).

Run:  python -m services.supervisor.server        (default port :8004)
Pair with ``bus_supervisor_enabled=true`` on the gateway so its simple timeout
sweep stands down.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from genie.messaging import get_awaiting_store
from genie.messaging.supervisor import Supervisor
from genie.observability import configure_logging, get_logger
from genie.platform.config import get_settings
from genie.platform.redis import redis_enabled

load_dotenv()
configure_logging()
_log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the ladder sweep + DLQ consumer; stop them on shutdown."""
    settings = get_settings()
    if not settings.kafka_enabled:
        raise RuntimeError("the Supervisor requires kafka_enabled=true (it is the async control plane)")
    if not redis_enabled():
        raise RuntimeError("the Supervisor requires Redis (redis_url) — dedup guards its produces")
    get_awaiting_store().ensure_indexes()
    supervisor = Supervisor()
    await supervisor.start()
    _log.info("supervisor.ready", extra={"attrs": {
        "max_extends": settings.bus_max_extends,
        "max_attempts": settings.bus_max_attempts,
    }})
    try:
        yield
    finally:
        await supervisor.stop()
        from genie.platform.db import close_all_connections

        await close_all_connections()


app = FastAPI(title="A2A Supervisor Service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "service": "a2a-supervisor"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=get_settings().bus_supervisor_port)
