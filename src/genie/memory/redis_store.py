"""Optional async Redis store: the hot blackboard mirror (working memory for an
active run, TTL'd). No-ops when REDIS_URL is unset or redis is missing, so the
framework runs without Redis."""

from __future__ import annotations

import json
import logging
from typing import Any

from genie.platform.redis import get_async_redis_client, redis_enabled

_log = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 3600  # 1h blackboard TTL (working memory for a run)


class RedisStore:
    """Thin async wrapper for blackboard hot-storage.

    No-ops when REDIS_URL is unset or the redis package is missing — keeps the
    framework usable for dev without standing up Redis.

    Connection management (including the per-event-loop client cache that keeps
    redis.asyncio clients from crossing loops) lives in :mod:`genie.platform.redis`;
    this store is just the blackboard domain logic on top.
    """

    def __init__(self) -> None:
        """Record whether Redis is available; connections come from the shared platform module."""
        self._enabled = redis_enabled()
        if not self._enabled:
            _log.warning("redis.disabled", extra={"attrs": {"reason": "REDIS_URL unset or redis package missing"}})

    @property
    def enabled(self) -> bool:
        """True only when REDIS_URL is set and the redis package imported."""
        return self._enabled

    def _client(self):
        """A redis client bound to the CURRENT running loop (from the shared platform module)."""
        return get_async_redis_client()

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


_store: RedisStore | None = None


def get_redis_store() -> RedisStore:
    """Return the process-wide RedisStore singleton, creating it on first use."""
    global _store
    if _store is None:
        _store = RedisStore()
    return _store
