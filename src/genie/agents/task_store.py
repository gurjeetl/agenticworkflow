"""In-process, TTL-bounded store of A2A :class:`Task` objects.

Each agent process keeps the Tasks it has produced so a client can later fetch
them with ``tasks/get`` (or cancel them with ``tasks/cancel``). Tasks are short-
lived request/reply artifacts here — agents run synchronously — so a small
in-memory map with a TTL sweep is sufficient; there is no cross-process sharing.

Thread-safe because the harness runs the agent on a worker thread while the
event loop may service another ``tasks/get`` concurrently.
"""
from __future__ import annotations

import threading
import time

from genie.a2a.types import Task


class TaskStore:
    """A thread-safe, TTL-expiring map of ``task_id -> Task``."""

    def __init__(self, ttl_seconds: float = 900.0) -> None:
        """Keep tasks for ``ttl_seconds`` (default 15 min) before they expire."""
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._tasks: dict[str, tuple[float, Task]] = {}

    def put(self, task: Task) -> None:
        """Store (or replace) a task, sweeping any expired entries first."""
        now = time.monotonic()
        with self._lock:
            self._sweep(now)
            self._tasks[task.id] = (now, task)

    def get(self, task_id: str) -> Task | None:
        """Return the live task for ``task_id``, or None if unknown/expired."""
        now = time.monotonic()
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                return None
            stored_at, task = entry
            if now - stored_at >= self._ttl:
                del self._tasks[task_id]
                return None
            return task

    def _sweep(self, now: float) -> None:
        """Drop expired entries. Caller must hold the lock."""
        expired = [tid for tid, (at, _) in self._tasks.items() if now - at >= self._ttl]
        for tid in expired:
            del self._tasks[tid]
