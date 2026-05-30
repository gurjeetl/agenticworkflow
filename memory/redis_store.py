from __future__ import annotations

import json
import logging
import os
from typing import Any

_log = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 86400  # 24h, matches MongoDB short-term TTL


class RedisStore:
    """Thin async wrapper for blackboard hot-storage.

    No-ops when REDIS_URL is unset or the redis package is missing —
    keeps the framework usable for dev without standing up Redis.
    """

    def __init__(self) -> None:
        self._url = os.getenv("REDIS_URL")
        self._client = None
        if not self._url:
            _log.warning("redis.disabled", extra={"attrs": {"reason": "REDIS_URL unset"}})
            return
        try:
            from redis import asyncio as redis_asyncio
        except ImportError:
            _log.warning("redis.disabled", extra={"attrs": {"reason": "redis package not installed"}})
            return
        self._client = redis_asyncio.from_url(self._url, encoding="utf-8", decode_responses=True)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def set_with_ttl(self, key: str, value: Any, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        if not self._client:
            return
        try:
            await self._client.set(key, json.dumps(value, default=str), ex=ttl_seconds)
        except Exception as e:
            _log.warning("redis.set_failed", extra={"attrs": {"key": key, "error": str(e)}})

    async def get(self, key: str) -> Any | None:
        if not self._client:
            return None
        try:
            raw = await self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            _log.warning("redis.get_failed", extra={"attrs": {"key": key, "error": str(e)}})
            return None

    async def delete_run(self, thread_id: str, run_id: str) -> None:
        if not self._client:
            return
        pattern = f"bb:{thread_id}:{run_id}:*"
        try:
            async for key in self._client.scan_iter(match=pattern):
                await self._client.delete(key)
        except Exception as e:
            _log.warning("redis.delete_run_failed", extra={"attrs": {"pattern": pattern, "error": str(e)}})

    async def close(self) -> None:
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass


_store: RedisStore | None = None


def get_redis_store() -> RedisStore:
    global _store
    if _store is None:
        _store = RedisStore()
    return _store
