"""Idempotent-consumption claims for the at-least-once bus (Redis SETNX).

Two distinct claim domains — a single per-cid key would wrongly reject retries,
which deliberately reuse the correlation id with ``attempt+1``:

* **inbox**: ``dedup:inbox:{agent_id}:{cid}:{attempt}`` — drops a *redelivered*
  attempt but lets a retry attempt through.
* **reply**: ``dedup:reply:{cid}`` — exactly one resume per correlation id;
  whichever of (real reply, timeout sweep, step.cancelled) claims first wins,
  late/duplicate replies are ignored.

Redis is REQUIRED in async mode — startup fails fast (see the bus consumers) —
so a missing client here is a hard error, never a silent pass-through.
"""
from __future__ import annotations

from genie.platform.config import get_settings
from genie.platform.redis import get_async_redis_client, redis_enabled


class Dedup:
    """SETNX-based claims; an injected client (tests) bypasses platform Redis."""

    def __init__(self, ttl_seconds: int | None = None, client=None) -> None:
        """TTL defaults to ``bus_dedup_ttl_seconds`` (≥ the longest deadline window)."""
        self._ttl = ttl_seconds or get_settings().bus_dedup_ttl_seconds
        self._client = client

    def _redis(self):
        """The injected client, else the platform per-loop async Redis client."""
        if self._client is not None:
            return self._client
        if not redis_enabled():
            raise RuntimeError("async A2A mode requires Redis (redis_url) for dedup")
        return get_async_redis_client()

    async def _claim(self, key: str) -> bool:
        """Atomically claim ``key``; True exactly once per TTL window."""
        return bool(await self._redis().set(key, "1", nx=True, ex=self._ttl))

    async def claim_inbox(self, agent_id: str, cid: str, attempt: str | int) -> bool:
        """Claim one delivery attempt on an agent's inbox."""
        return await self._claim(f"dedup:inbox:{agent_id}:{cid}:{attempt}")

    async def claim_reply(self, cid: str) -> bool:
        """Claim the single resume slot for a correlation id."""
        return await self._claim(f"dedup:reply:{cid}")

    async def claim_group_resume(self, group_id: str) -> bool:
        """Claim the single graph-resume slot for a fan-out group (one wave)."""
        return await self._claim(f"dedup:resume:{group_id}")
