"""Plan/Subtask DAG models shared by the Planner, Orchestrator, and Executor.

Defines the validated subtask graph and the wave decomposition (Kahn's
algorithm) that orders execution by dependency.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DAGCycleError(ValueError):
    """Raised when the subtask dependency graph contains a cycle (cannot be waved)."""
    pass


class Subtask(BaseModel):
    """One agent invocation in the plan: which agent, its args, and its dependencies."""

    id: str
    agent_id: str
    agent_version: str = "1.0.0"
    args: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    sla_ms: int = 10000


class Plan(BaseModel):
    """The full subtask DAG for one request, plus its wave decomposition."""

    subtasks: list[Subtask] = Field(default_factory=list)

    def by_id(self) -> dict[str, Subtask]:
        """Index subtasks by id for dependency lookups."""
        return {t.id: t for t in self.subtasks}

    def waves(self) -> list[list[Subtask]]:
        """Kahn's algorithm: group subtasks into dependency levels (waves).

        All tasks in wave N have all their deps satisfied by tasks in waves < N.
        Raises DAGCycleError if the graph contains a cycle.
        """
        by_id = self.by_id()
        for t in self.subtasks:
            for dep in t.depends_on:
                if dep not in by_id:
                    raise ValueError(f"subtask '{t.id}' depends on unknown '{dep}'")

        remaining = {t.id: set(t.depends_on) for t in self.subtasks}
        levels: list[list[Subtask]] = []
        while remaining:
            ready = [tid for tid, deps in remaining.items() if not deps]
            if not ready:
                raise DAGCycleError(f"cycle detected among: {sorted(remaining)}")
            levels.append([by_id[tid] for tid in ready])
            for tid in ready:
                remaining.pop(tid)
            for deps in remaining.values():
                deps.difference_update(ready)
        return levels
