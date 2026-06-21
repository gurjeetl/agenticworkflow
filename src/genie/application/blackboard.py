"""The per-run shared workspace agents read from and write results into.

Backs ``state["blackboard"]``: the Executor writes each task's result here and
the Gate/Synthesizer read it. Mirrors writes to Redis for cross-process visibility.
"""
from __future__ import annotations

from typing import Any

from genie.memory.redis_store import get_redis_store


class Blackboard:
    """Shared workspace for one run.

    Two-layer write: an in-memory dict mirrored on AgentState (so downstream
    LangGraph nodes see it synchronously) plus Redis for cross-process / audit
    visibility. Redis is best-effort — failures are logged and ignored.
    """

    def __init__(self, thread_id: str, run_id: str) -> None:
        self.thread_id = thread_id
        self.run_id = run_id
        self._mem: dict[str, dict] = {}
        self._redis = get_redis_store()

    def snapshot(self) -> dict[str, dict]:
        """Shallow copy of the in-memory entries, for mirroring onto AgentState."""
        return dict(self._mem)

    def get(self, task_id: str) -> dict | None:
        """Return a task's written entry, or None if it hasn't produced one yet."""
        return self._mem.get(task_id)

    async def write(self, task_id: str, payload: dict[str, Any]) -> None:
        """Record a task's result in memory and best-effort mirror it to Redis."""
        self._mem[task_id] = payload
        if self._redis.enabled:
            key = f"bb:{self.thread_id}:{self.run_id}:{task_id}"
            await self._redis.set_with_ttl(key, payload)

    async def write_error(self, task_id: str, message: str) -> None:
        """Record a task failure as an ``{"error": ...}`` entry the gate can detect."""
        await self.write(task_id, {"error": message})

    def has_errors(self) -> bool:
        """True if any task wrote an error entry this run."""
        return any("error" in entry for entry in self._mem.values())

    def error_keys(self) -> list[str]:
        """Task ids whose entries carry an error."""
        return [tid for tid, entry in self._mem.items() if "error" in entry]
