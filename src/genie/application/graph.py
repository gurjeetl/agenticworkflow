"""LangGraph wiring for the multi-agent pipeline.

Builds and compiles the StateGraph that threads an ``AgentState`` through the
nodes in order: input_guard → router → (fast | chitchat | full plan) →
synthesizer → output_guard. The conditional-edge routers below pick the next
node from state set by the preceding node.
"""
import mlflow
from langgraph.graph import StateGraph, START, END

from genie.application.nodes.completion_gate import CompletionGate
from genie.application.checkpointer import create_memory
from genie.application.nodes.executor import Executor
from genie.application.nodes.orchestrator import Orchestrator
from genie.application.nodes.planner import PlannerAgent
from genie.application.nodes.router import RouterAgent
from genie.security import InputGuard, OutputGuard
from genie.application.state import AgentState
from genie.application.nodes.synthesizer import SynthesizerAgent
from genie.platform.config import get_settings

# Agents no longer run in-process: each runs as its own service and self-registers
# with the Registry Service. The Planner discovers them and the Executor invokes
# them over HTTP, so there is nothing to import here for registration.


def route_after_input_guard(state: AgentState) -> str:
    """Send a guard-blocked prompt straight to END, else continue into the pipeline.

    Returns the neutral ``"continue"`` key; build_graph maps it to the Router when
    the Router is enabled, or directly to the Planner when it is disabled.
    """
    decision = "blocked" if state.get("guard_block") else "continue"

    span = mlflow.get_current_active_span()
    if span is not None:
        try:
            span.add_event("input_guard.routing", attributes={"next": decision})
        except Exception:
            pass
    return decision


def route_after_router(state: AgentState) -> str:
    """Map the Router's route to its node: fast→executor, chitchat→synthesizer, else planner."""
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
    """Loop back to the Planner on a replan decision, else converge on the Synthesizer."""
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
    """Construct, wire, and compile the pipeline StateGraph with an in-memory checkpointer.

    The Router triage node is optional (``settings.router_enabled``): when disabled
    it is omitted from the graph and the input guard hands off directly to the
    Planner, so every request runs the full planning pipeline.

    The content guard is likewise optional (``settings.llm_guard_enabled``, ON by
    default): when disabled BOTH guard nodes are omitted and the llm-guard models
    are never loaded, so the prompt enters — and the answer leaves — the pipeline
    unscanned.
    """
    settings = get_settings()
    router_enabled = settings.router_enabled
    guard_enabled = settings.llm_guard_enabled

    planner = PlannerAgent()
    orchestrator = Orchestrator()
    executor = Executor()
    gate = CompletionGate()
    synthesizer = SynthesizerAgent()

    graph = StateGraph(AgentState)

    if guard_enabled:
        # Construct the guards only when enabled — they load the local llm-guard
        # models, so skipping them avoids that cost entirely when disabled.
        graph.add_node("input_guard", InputGuard().run)
        graph.add_node("output_guard", OutputGuard().run)
    if router_enabled:
        # Construct the Router only when enabled — it loads a local intent
        # classifier, so skipping it avoids that cost entirely when disabled.
        router = RouterAgent()
        graph.add_node("router", router.run)
    graph.add_node("planner", planner.run)
    graph.add_node("orchestrator", orchestrator.run)
    graph.add_node("executor", executor.run)
    graph.add_node("gate", gate.run)
    graph.add_node("synthesizer", synthesizer.run)

    # First pipeline node after the (optional) input guard: the Router when
    # enabled — which fast-paths to the executor, sends chitchat to the
    # synthesizer, or falls through to the full planner — else the Planner.
    first_node = "router" if router_enabled else "planner"

    # Input guard scans the user prompt first when enabled. Blocked → straight to
    # END with a safe refusal; otherwise the (PII-redacted) prompt flows on. With
    # the guard disabled the raw prompt enters the pipeline directly.
    if guard_enabled:
        graph.add_edge(START, "input_guard")
        graph.add_conditional_edges(
            "input_guard",
            route_after_input_guard,
            {
                "continue": first_node,
                "blocked": END,
            },
        )
    else:
        graph.add_edge(START, first_node)
    if router_enabled:
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
    # answer before it reaches the user when enabled, else the answer is returned
    # straight from the synthesizer.
    if guard_enabled:
        graph.add_edge("synthesizer", "output_guard")
        graph.add_edge("output_guard", END)
    else:
        graph.add_edge("synthesizer", END)

    memory = create_memory()
    return graph.compile(checkpointer=memory)


_graph = None


def get_graph():
    """Return the process-wide compiled graph, building it once on first use."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
