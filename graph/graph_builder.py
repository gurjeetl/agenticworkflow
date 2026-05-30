import mlflow
from langgraph.graph import StateGraph, START, END

from gate import CompletionGate
from memory.memory import create_memory
from orchestrator import Orchestrator
from planner import PlannerAgent
from state import AgentState
from synthesizer import SynthesizerAgent

# Importing the agents package side-effect: each agent module registers itself
# in the AGENT_REGISTRY at import time.
import agents.weather_agent  # noqa: F401
import agents.outage_agent  # noqa: F401


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
    planner = PlannerAgent()
    orchestrator = Orchestrator()
    gate = CompletionGate()
    synthesizer = SynthesizerAgent()

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner.run)
    graph.add_node("orchestrator", orchestrator.run)
    graph.add_node("gate", gate.run)
    graph.add_node("synthesizer", synthesizer.run)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "orchestrator")
    graph.add_edge("orchestrator", "gate")
    graph.add_conditional_edges(
        "gate",
        route_after_gate,
        {
            "planner": "planner",
            "synthesizer": "synthesizer",
        },
    )
    graph.add_edge("synthesizer", END)

    memory = create_memory()
    return graph.compile(checkpointer=memory)
