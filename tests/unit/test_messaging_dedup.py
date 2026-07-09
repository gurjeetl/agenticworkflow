"""Unit tests for the two dedup domains (inbox per-attempt, reply per-cid)."""
from __future__ import annotations

from genie.messaging.dedup import Dedup


class StubRedis:
    """Minimal async SETNX: True on first claim of a key, None afterwards."""

    def __init__(self) -> None:
        self.keys: dict[str, str] = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.keys:
            return None
        self.keys[key] = value
        return True


async def test_inbox_dedup_drops_redelivery_but_allows_retry():
    dedup = Dedup(ttl_seconds=60, client=StubRedis())
    assert await dedup.claim_inbox("weather", "cid-1", 1) is True
    # Redelivered same attempt (Kafka at-least-once) → dropped.
    assert await dedup.claim_inbox("weather", "cid-1", 1) is False
    # Retry keeps the cid but bumps attempt → passes. A single per-cid key
    # would wrongly reject this — the exact bug the two-domain design fixes.
    assert await dedup.claim_inbox("weather", "cid-1", 2) is True


async def test_reply_dedup_allows_exactly_one_resume():
    dedup = Dedup(ttl_seconds=60, client=StubRedis())
    assert await dedup.claim_reply("cid-1") is True
    # Late/duplicate reply, or the deadline sweep racing the real reply.
    assert await dedup.claim_reply("cid-1") is False


async def test_dedup_domains_are_independent():
    dedup = Dedup(ttl_seconds=60, client=StubRedis())
    assert await dedup.claim_inbox("weather", "cid-1", 1) is True
    assert await dedup.claim_reply("cid-1") is True  # different key space
