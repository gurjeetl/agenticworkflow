"""Centralized Redis connections for the platform.

Returns ``None`` everywhere when ``redis_url`` is unset or the redis package is
missing, so the framework runs without Redis (callers fail open).

The async client is cached **per event loop**: redis.asyncio connections are bound
to the loop that created them, and the platform runs more than one loop (a transient
``_run_async`` loop in the Executor plus the gateway's main loop). Sharing one client
across loops raises "attached to a different loop" / stale-connection errors, so we
keep one client per loop. A separate process-wide sync client is offered for sync
callers (agents/tools/LangGraph nodes) since redis-py supports sync too.
"""
from __future__ import annotations

import asyncio
import logging

from genie.platform.config import get_settings

_log = logging.getLogger(__name__)

# event loop -> redis.asyncio client
_async_clients: dict = {}
_sync_client = None


def redis_enabled() -> bool:
    """True only when redis_url is configured and the redis package is importable."""
    if not get_settings().redis_url:
        return False
    try:
        import redis  # noqa: F401
    except ImportError:
        return False
    return True


def get_async_redis_client():
    """Return an async client bound to the CURRENT running loop (created on demand).

    Returns ``None`` when Redis is unconfigured/unavailable or no loop is running.
    """
    if not redis_enabled():
        return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    # Drop clients whose loop has closed so the dict doesn't grow unbounded.
    for dead in [lp for lp in _async_clients if lp.is_closed()]:
        _async_clients.pop(dead, None)
    client = _async_clients.get(loop)
    if client is None:
        from redis import asyncio as redis_asyncio
        # protocol=2 (RESP2): redis-py 8 defaults to RESP3 and sends `HELLO 3` on
        # connect, which Redis < 6 rejects. RESP2 works on all versions.
        client = redis_asyncio.from_url(
            get_settings().redis_url, encoding="utf-8", decode_responses=True, protocol=2
        )
        _async_clients[loop] = client
    return client


def get_sync_redis_client():
    """Return the process-wide sync redis client, or ``None`` when unavailable."""
    global _sync_client
    if not redis_enabled():
        return None
    if _sync_client is None:
        import redis
        _sync_client = redis.Redis.from_url(
            get_settings().redis_url, encoding="utf-8", decode_responses=True, protocol=2
        )
    return _sync_client


async def close_redis_clients() -> None:
    """Close every per-loop async client and the sync client (errors swallowed)."""
    global _sync_client
    for client in list(_async_clients.values()):
        try:
            await client.close()
        except Exception:
            pass
    _async_clients.clear()
    if _sync_client is not None:
        try:
            _sync_client.close()
        except Exception:
            pass
        _sync_client = None
