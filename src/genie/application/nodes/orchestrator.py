"""Orchestrator node: turns the Planner's DAG into ordered dependency waves.

Pure decomposition — runs no agents. Computes the execution waves the Executor
consumes, keeping planning and execution observable as separate trace spans.
"""
from __future__ import annotations

from mlflow.entities import SpanType

from genie.agents.base import patch
from genie.observability import Observable
from genie.application.nodes._planner_dag import Plan
from genie.application.state import AgentState


class Orchestrator(Observable):
    """Decomposes the plan's DAG into dependency waves.

    This node does *not* run any agents — it computes the execution waves
    (Kahn's algorithm), enforces plan-level validity, and hands the
    decomposition to the Executor node. Splitting decomposition from
    execution makes each phase observable on its own in the trace.
    """

    _traced_methods: tuple[str, ...] = ("run",)
    _component_kind: str = "orchestrator"
    _span_type: str = SpanType.CHAIN

    def run(self, state: AgentState) -> AgentState:
        """Compute wave ids from the plan; surface a cycle/invalid-DAG error to the Executor."""
        plan_dict = state.get("plan") or {}
        plan = Plan(**plan_dict)

        if not plan.subtasks:
            self.log_event("orchestrator.empty_plan")
            return patch(state, waves=[], plan_error=None, wave_cursor=0)

        try:
            waves = plan.waves()
        except Exception as e:
            self.log("error", "orchestrator.dag_invalid", error=str(e))
            return patch(state, waves=[], plan_error=str(e), wave_cursor=0)

        wave_ids = [[t.id for t in wave] for wave in waves]
        self.log_event(
            "orchestrator.decomposed",
            wave_count=len(wave_ids),
            task_count=sum(len(w) for w in wave_ids),
        )
        # wave_cursor=0 restarts the Executor's wave loop for this (re)plan.
        return patch(state, waves=wave_ids, plan_error=None, wave_cursor=0)
