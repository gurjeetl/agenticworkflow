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

    # Router decision: "plan" (full pipeline) | "fast" (skip to executor) | "chitchat" (skip to synthesizer)
    route: Optional[str]

    # Planner / Orchestrator / Executor / Gate
    plan: Optional[dict]
    agent_versions: dict[str, str]
    waves: Optional[list[list[str]]]   # Orchestrator decomposition: task ids per wave
    plan_error: Optional[str]          # DAG decomposition failure, surfaced to Executor
    blackboard: dict[str, dict]
    blackboard_snapshot: Optional[dict]
    replan_count: int
    max_replans: int
    replan_reason: Optional[str]
    partial: bool

    # Per-node real database operations, surfaced to the Tracer's Live DB State
    # panel. Each node overwrites this with the ops it performed this step:
    # [{store: "redis"|"mongodb"|"milvus", op: "read"|"write"|"search",
    #   detail, code?, keys?, enabled}]. Nodes with no store I/O leave it unset.
    db_ops: Optional[list[dict]]

    # LLM Guard (mandatory content guard): set by the input/output guard nodes.
    # guard_block is truthy only when a blocking scanner fired (short-circuits to
    # END with a safe refusal); guard_input/guard_output record each scan for the
    # tracer.
    guard_block: Optional[dict]
    guard_input: Optional[dict]
    guard_output: Optional[dict]

    # Output & control
    final_output: Optional[str]
    view: Optional[dict]
    is_complete: bool
    error: Optional[str]
