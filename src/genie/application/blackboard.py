"""The per-run shared workspace agents read from and write results into.

Backs ``state["blackboard"]``: the Executor writes each task's result here and
the Gate/Synthesizer read it. Mirrors writes to Redis for cross-process
visibility, and — when Redis is configured — **reads through** to the mirror on
a local miss, so a downstream consumer in another process can pick up an
upstream result "when ready" (the diagram's implicit-coordination arrow).

Keys are tenant-scoped when a tenant is present:
``bb:{tenant}:{thread}:{run}:{task}``; without a tenant the pre-multi-tenancy
``bb:{thread}:{run}:{task}`` shape is preserved (backward compatible).
"""
from __future__ import annotations

import json
from typing import Any

from genie.memory.redis_store import get_redis_store
from genie.platform.redis import get_sync_redis_client, redis_enabled


class Blackboard:
    """Shared workspace for one run.

    Two-layer write: an in-memory dict mirrored on AgentState (so downstream
    LangGraph nodes see it synchronously) plus Redis for cross-process / audit
    visibility. Redis is best-effort — failures are logged and ignored.
    """

    def __init__(self, thread_id: str, run_id: str, tenant_id: str | None = None) -> None:
        """Set up the empty in-memory workspace for one run and grab the Redis mirror store."""
        self.thread_id = thread_id
        self.run_id = run_id
        self.tenant_id = tenant_id or None
        self._mem: dict[str, dict] = {}
        self._redis = get_redis_store()

    def key(self, task_id: str) -> str:
        """The Redis mirror key for one task (tenant-scoped when a tenant is set)."""
        if self.tenant_id:
            return f"bb:{self.tenant_id}:{self.thread_id}:{self.run_id}:{task_id}"
        return f"bb:{self.thread_id}:{self.run_id}:{task_id}"

    def snapshot(self) -> dict[str, dict]:
        """Shallow copy of the in-memory entries, for mirroring onto AgentState."""
        return dict(self._mem)

    def get(self, task_id: str) -> dict | None:
        """A task's entry: local memory first, then read-through to the Redis mirror.

        The read-through is what lets a process that did NOT produce an entry
        (another gateway instance, a downstream consumer) still resolve it.
        Sync client on purpose — ``get`` is called from synchronous node code.
        """
        entry = self._mem.get(task_id)
        if entry is not None:
            return entry
        if not redis_enabled():
            return None
        try:
            client = get_sync_redis_client()
            raw = client.get(self.key(task_id)) if client else None
            if raw:
                entry = json.loads(raw)
                self._mem[task_id] = entry
                return entry
        except Exception:
            pass  # mirror is best-effort in both directions
        return None

    async def write(self, task_id: str, payload: dict[str, Any]) -> None:
        """Record a task's result in memory and best-effort mirror it to Redis."""
        self._mem[task_id] = payload
        if self._redis.enabled:
            await self._redis.set_with_ttl(self.key(task_id), payload)

    async def write_error(self, task_id: str, message: str) -> None:
        """Record a task failure as an ``{"error": ...}`` entry the gate can detect."""
        await self.write(task_id, {"error": message})

    def has_errors(self) -> bool:
        """True if any task wrote an error entry this run."""
        return any("error" in entry for entry in self._mem.values())

    def error_keys(self) -> list[str]:
        """Task ids whose entries carry an error."""
        return [tid for tid, entry in self._mem.items() if "error" in entry]
