from typing import TypedDict, Optional, Annotated
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    # Input
    user_input: str
    current_task: str
    thread_id: str
    run_id: str

    # Conversation
    messages: Annotated[list[BaseMessage], operator.add]

    # Agent reasoning
    agent_scratchpad: str

    # Loop control
    iteration_count: int
    max_iterations: int

    # Tool use (future-ready)
    tool_calls: list[dict]
    tool_results: list[dict]

    # Memory
    short_term_memory: list[str]
    long_term_memory_keys: list[str]

    # Supervisor routing (legacy; kept for backwards-compat during migration)
    active_agent: str
    next_action: str
    delegated_task: Optional[str]

    # Domain (legacy; planner ignores these — agents now read from blackboard args)
    location: Optional[str]
    intent: Optional[str]
    outage_id: Optional[int]

    # Planner / Orchestrator / Gate
    plan: Optional[dict]
    agent_versions: dict[str, str]
    blackboard: dict[str, dict]
    blackboard_snapshot: Optional[dict]
    replan_count: int
    max_replans: int
    replan_reason: Optional[str]
    partial: bool

    # Output & control
    final_output: Optional[str]
    view: Optional[dict]
    is_complete: bool
    error: Optional[str]
