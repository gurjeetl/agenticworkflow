from __future__ import annotations

from typing import Any

from memory.redis_store import get_redis_store


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
        return dict(self._mem)

    def get(self, task_id: str) -> dict | None:
        return self._mem.get(task_id)

    async def write(self, task_id: str, payload: dict[str, Any]) -> None:
        self._mem[task_id] = payload
        if self._redis.enabled:
            key = f"bb:{self.thread_id}:{self.run_id}:{task_id}"
            await self._redis.set_with_ttl(key, payload)

    async def write_error(self, task_id: str, message: str) -> None:
        await self.write(task_id, {"error": message})

    def has_errors(self) -> bool:
        return any("error" in entry for entry in self._mem.values())

    def error_keys(self) -> list[str]:
        return [tid for tid, entry in self._mem.items() if "error" in entry]
