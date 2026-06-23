"""Optional async Redis store: the hot blackboard mirror (working memory for an
active run, TTL'd). No-ops when REDIS_URL is unset or redis is missing, so the
framework runs without Redis."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from genie.platform.config import get_settings

_log = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 3600  # 1h blackboard TTL (working memory for a run)


class RedisStore:
    """Thin async wrapper for blackboard hot-storage.

    No-ops when REDIS_URL is unset or the redis package is missing — keeps the
    framework usable for dev without standing up Redis.

    Loop-aware: redis.asyncio connections are bound to the event loop that
    created them. The Executor mirrors the blackboard from a transient
    ``_run_async`` loop while the FastAPI handlers read/delete on the main loop —
    sharing one client across loops raises "attached to a different loop" /
    stale-connection errors (which silently dropped writes and made
    delete_run/get_run flaky). We therefore keep one client per event loop.
    """

    def __init__(self) -> None:
        """Read REDIS_URL and prepare a per-event-loop client cache; stay disabled if unset or the redis package is unavailable."""
        self._url = get_settings().redis_url
        self._enabled = False
        self._clients: dict = {}  # event loop -> redis client
        if not self._url:
            _log.warning("redis.disabled", extra={"attrs": {"reason": "REDIS_URL unset"}})
            return
        try:
            from redis import asyncio as redis_asyncio  # noqa: F401
        except ImportError:
            _log.warning("redis.disabled", extra={"attrs": {"reason": "redis package not installed"}})
            return
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """True only when REDIS_URL is set and the redis package imported."""
        return self._enabled

    def _client(self):
        """A redis client bound to the CURRENT running loop (created on demand)."""
        if not self._enabled:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        # Drop clients whose loop has closed so the dict doesn't grow unbounded.
        for dead in [l for l in self._clients if l.is_closed()]:
            self._clients.pop(dead, None)
        client = self._clients.get(loop)
        if client is None:
            from redis import asyncio as redis_asyncio
            # protocol=2 (RESP2): redis-py 8 defaults to RESP3 and sends `HELLO 3`
            # on connect, which Redis < 6 rejects. RESP2 works on all versions.
            client = redis_asyncio.from_url(
                self._url, encoding="utf-8", decode_responses=True, protocol=2
            )
            self._clients[loop] = client
        return client

    async def set_with_ttl(self, key: str, value: Any, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        """Write a JSON-encoded value with a TTL. No-op/fail-open when disabled."""
        client = self._client()
        if not client:
            return
        try:
            await client.set(key, json.dumps(value, default=str), ex=ttl_seconds)
        except Exception as e:
            _log.warning("redis.set_failed", extra={"attrs": {"key": key, "error": str(e)}})

    async def get(self, key: str) -> Any | None:
        """Read and JSON-decode a value. Returns None when missing or disabled."""
        client = self._client()
        if not client:
            return None
        try:
            raw = await client.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            _log.warning("redis.get_failed", extra={"attrs": {"key": key, "error": str(e)}})
            return None

    async def get_run(self, thread_id: str, run_id: str) -> dict[str, Any]:
        """Return all blackboard entries for one run as {task_id: payload}.

        Reads back the mirror written by Blackboard.write. Returns {} when Redis
        is disabled. task_id is recovered by stripping the literal key prefix
        (robust to thread_ids that themselves contain ':', e.g. trace threads).
        """
        client = self._client()
        if not client:
            return {}
        prefix = f"bb:{thread_id}:{run_id}:"
        entries: dict[str, Any] = {}
        try:
            async for key in client.scan_iter(match=f"{prefix}*"):
                task_id = key[len(prefix):]
                raw = await client.get(key)
                if raw:
                    entries[task_id] = json.loads(raw)
        except Exception as e:
            _log.warning("redis.get_run_failed", extra={"attrs": {"prefix": prefix, "error": str(e)}})
        return entries

    async def delete_run(self, thread_id: str, run_id: str) -> None:
        """Delete all blackboard mirror entries for one run. No-op when disabled."""
        client = self._client()
        if not client:
            return
        pattern = f"bb:{thread_id}:{run_id}:*"
        try:
            async for key in client.scan_iter(match=pattern):
                await client.delete(key)
        except Exception as e:
            _log.warning("redis.delete_run_failed", extra={"attrs": {"pattern": pattern, "error": str(e)}})

    async def close(self) -> None:
        """Close every per-loop client and clear the cache (errors swallowed)."""
        for client in list(self._clients.values()):
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()


_store: RedisStore | None = None


def get_redis_store() -> RedisStore:
    """Return the process-wide RedisStore singleton, creating it on first use."""
    global _store
    if _store is None:
        _store = RedisStore()
    return _store
