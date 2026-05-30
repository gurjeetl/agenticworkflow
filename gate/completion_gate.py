from __future__ import annotations

from mlflow.entities import SpanType

from baseagent.base_agent import patch
from observability import Observable
from planner.dag import Plan
from state import AgentState


class CompletionGate(Observable):
    """Decides: synthesize the final answer, or re-plan?"""

    _traced_methods: tuple[str, ...] = ("run",)
    _component_kind: str = "gate"
    _span_type: str = SpanType.CHAIN

    def run(self, state: AgentState) -> AgentState:
        plan_dict = state.get("plan") or {}
        plan = Plan(**plan_dict)
        blackboard = state.get("blackboard") or {}
        replan_count = state.get("replan_count", 0) or 0
        max_replans = state.get("max_replans", 3) or 3

        all_present = all(t.id in blackboard for t in plan.subtasks)
        error_keys = [tid for tid, entry in blackboard.items() if isinstance(entry, dict) and "error" in entry]
        partial = bool(error_keys)

        self.log_event(
            "gate.inspect",
            plan_size=len(plan.subtasks),
            blackboard_size=len(blackboard),
            error_count=len(error_keys),
            replan_count=replan_count,
            max_replans=max_replans,
            all_present=all_present,
        )

        budget_left = replan_count < max_replans
        empty_plan = len(plan.subtasks) == 0
        should_replan = (not empty_plan) and (not all_present or partial) and budget_left

        if should_replan:
            return patch(
                state,
                next_action="replan",
                replan_count=replan_count + 1,
                replan_reason=(
                    f"missing tasks: {[t.id for t in plan.subtasks if t.id not in blackboard]}; "
                    f"errored tasks: {error_keys}"
                ),
                blackboard_snapshot=dict(blackboard),
                partial=partial,
            )

        return patch(
            state,
            next_action="synthesize",
            partial=partial,
        )
