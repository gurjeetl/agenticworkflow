"""Shared construction of the per-task AgentState.

Both the (out-of-process) agent server harness and any in-process caller build the
exact same narrow ``AgentState`` for a single subtask: a clean state seeded with
the task's identifiers, the blackboard snapshot for cross-task reads, and the
task's args spread as top-level keys (existing agents read ``state["location"]``,
``state["outage_id"]``, etc.). Keeping it in one place guarantees the remote
invocation path can never drift from the original in-process behavior.
"""
from __future__ import annotations

from state import AgentState


def build_task_state(
    *,
    task_id: str,
    agent_id: str,
    args: dict | None,
    thread_id: str,
    run_id: str,
    blackboard: dict | None,
) -> AgentState:
    task_state: AgentState = {
        "user_input": "",
        "current_task": task_id,
        "thread_id": thread_id,
        "run_id": run_id,
        "messages": [],
        "agent_scratchpad": "",
        "iteration_count": 0,
        "max_iterations": 5,
        "tool_calls": [],
        "tool_results": [],
        "short_term_memory": [],
        "long_term_memory_keys": [],
        "active_agent": agent_id,
        "next_action": "",
        "delegated_task": None,
        "location": None,
        "intent": None,
        "outage_id": None,
        "plan": None,
        "agent_versions": {},
        "waves": None,
        "plan_error": None,
        "blackboard": blackboard or {},
        "blackboard_snapshot": None,
        "replan_count": 0,
        "max_replans": 0,
        "replan_reason": None,
        "partial": False,
        "final_output": None,
        "view": None,
        "is_complete": False,
        "error": None,
    }
    for k, v in (args or {}).items():
        task_state[k] = v  # type: ignore[literal-required]
    return task_state
