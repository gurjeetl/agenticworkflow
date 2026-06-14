import mlflow
from langgraph.graph import StateGraph, START, END

from gate import CompletionGate
from memory.memory import create_memory
from orchestrator import Executor, Orchestrator
from planner import PlannerAgent
from router import RouterAgent
from security import InputGuard, OutputGuard
from state import AgentState
from synthesizer import SynthesizerAgent

# Agents no longer run in-process: each runs as its own service and self-registers
# with the Registry Service. The Planner discovers them and the Executor invokes
# them over HTTP, so there is nothing to import here for registration.


def route_after_input_guard(state: AgentState) -> str:
    decision = "blocked" if state.get("guard_block") else "router"

    span = mlflow.get_current_active_span()
    if span is not None:
        try:
            span.add_event("input_guard.routing", attributes={"next": decision})
        except Exception:
            pass
    return decision


def route_after_router(state: AgentState) -> str:
    route = state.get("route") or "plan"
    decision = {"fast": "executor", "chitchat": "synthesizer"}.get(route, "planner")

    span = mlflow.get_current_active_span()
    if span is not None:
        try:
            span.add_event("router.routing", attributes={"route": route, "next": decision})
        except Exception:
            pass
    return decision


def route_after_gate(state: AgentState) -> str:
    action = state.get("next_action") or "synthesize"
    decision = "planner" if action == "replan" else "synthesizer"

    span = mlflow.get_current_active_span()
    if span is not None:
        try:
            span.add_event("gate.routing", attributes={
                "next": decision,
                "replan_count": state.get("replan_count", 0),
                "max_replans": state.get("max_replans", 3),
                "partial": bool(state.get("partial")),
            })
        except Exception:
            pass
    return decision


def build_graph():
    input_guard = InputGuard()
    router = RouterAgent()
    planner = PlannerAgent()
    orchestrator = Orchestrator()
    executor = Executor()
    gate = CompletionGate()
    synthesizer = SynthesizerAgent()
    output_guard = OutputGuard()

    graph = StateGraph(AgentState)

    graph.add_node("input_guard", input_guard.run)
    graph.add_node("router", router.run)
    graph.add_node("planner", planner.run)
    graph.add_node("orchestrator", orchestrator.run)
    graph.add_node("executor", executor.run)
    graph.add_node("gate", gate.run)
    graph.add_node("synthesizer", synthesizer.run)
    graph.add_node("output_guard", output_guard.run)

    # Input guard scans the user prompt first. Blocked → straight to END with a
    # safe refusal; otherwise the (PII-redacted) prompt flows to the Router, which
    # fast-paths to the executor, sends chitchat to the synthesizer, or falls
    # through to the full planner pipeline.
    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges(
        "input_guard",
        route_after_input_guard,
        {
            "router": "router",
            "blocked": END,
        },
    )
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "planner": "planner",
            "executor": "executor",
            "synthesizer": "synthesizer",
        },
    )
    graph.add_edge("planner", "orchestrator")
    graph.add_edge("orchestrator", "executor")
    graph.add_edge("executor", "gate")
    graph.add_conditional_edges(
        "gate",
        route_after_gate,
        {
            "planner": "planner",
            "synthesizer": "synthesizer",
        },
    )
    # Every route converges at the synthesizer; the output guard scans the final
    # answer before it reaches the user.
    graph.add_edge("synthesizer", "output_guard")
    graph.add_edge("output_guard", END)

    memory = create_memory()
    return graph.compile(checkpointer=memory)
