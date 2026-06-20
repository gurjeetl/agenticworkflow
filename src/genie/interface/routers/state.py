"""Introspection endpoints: graph checkpoint state and the Redis blackboard mirror."""
from fastapi import APIRouter

from genie.application.checkpointer import get_thread_config
from genie.application.graph import get_graph
from genie.memory.redis_store import get_redis_store

router = APIRouter()


@router.get("/state/{thread_id}")
async def get_state(thread_id: str):
    config = get_thread_config(thread_id)
    snapshot = get_graph().get_state(config)
    return snapshot.values


@router.get("/blackboard/{thread_id}/{run_id}")
async def get_blackboard(thread_id: str, run_id: str):
    """Read back the Redis-mirrored blackboard entries for one run.

    Returns {"enabled": false, "entries": {}} when Redis is disabled (REDIS_URL
    unset or the redis package missing) — the blackboard mirror is best-effort.
    """
    store = get_redis_store()
    return {"enabled": store.enabled, "entries": await store.get_run(thread_id, run_id)}
