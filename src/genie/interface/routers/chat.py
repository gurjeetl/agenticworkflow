"""Chat endpoints: run the full graph (/chat) and a step-by-step trace (/chat/trace)."""
import os
import time
import uuid

import mlflow
from fastapi import APIRouter
from langchain_core.messages import AIMessage, HumanMessage
from mlflow.entities import SpanType
from pydantic import BaseModel

from genie.application.checkpointer import get_thread_config
from genie.application.graph import get_graph
from genie.memory.mongo_store import get_mongo_store
from genie.memory.redis_store import get_redis_store
from genie.observability import get_logger

router = APIRouter()
_log = get_logger(__name__)

# Graph nodes that actually perform a store operation (and emit db_ops).
_DB_OP_PRODUCERS = {"planner", "executor", "synthesizer"}


class ChatRequest(BaseModel):
    message: str
    thread_id: str


def _base_state(message: str, thread_id: str, run_id: str, prior_messages, long_term_keys, prior_values=None) -> dict:
    prior_values = prior_values or {}
    return {
        "user_input": message,
        "current_task": "",
        "thread_id": thread_id,
        "run_id": run_id,
        "messages": prior_messages + [HumanMessage(content=message)],
        "agent_scratchpad": "",
        "iteration_count": 0,
        "max_iterations": 10,
        "tool_calls": [],
        "tool_results": [],
        "short_term_memory": [],
        "long_term_memory_keys": long_term_keys,
        "active_agent": "",
        "next_action": "",
        "delegated_task": None,
        "location": prior_values.get("location"),
        "intent": prior_values.get("intent"),
        "outage_id": None,
        "route": None,
        "plan": None,
        "agent_versions": {},
        "waves": None,
        "plan_error": None,
        "blackboard": {},
        "blackboard_snapshot": None,
        "replan_count": 0,
        "max_replans": 3,
        "replan_reason": None,
        "partial": False,
        "guard_block": None,
        "guard_input": None,
        "guard_output": None,
        "final_output": None,
        "view": None,
        "is_complete": False,
        "error": None,
        "db_ops": None,
    }


@router.post("/chat")
async def chat(req: ChatRequest):
    graph = get_graph()
    with mlflow.start_span(name="chat.request", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"thread_id": req.thread_id, "message_length": len(req.message)})
        _log.info("chat.request", extra={"attrs": {"thread_id": req.thread_id, "message_length": len(req.message)}})

        store = get_mongo_store()
        prior_messages = await store.get_messages(req.thread_id)
        facts = await store.get_facts(req.thread_id)
        long_term_keys = [f"{k}: {v}" for k, v in facts.items()]

        config = get_thread_config(req.thread_id)
        prior_snapshot = graph.get_state(config)
        prior_values = prior_snapshot.values if prior_snapshot and prior_snapshot.values else {}
        run_id = uuid.uuid4().hex
        span.set_attribute("run_id", run_id)
        state = _base_state(req.message, req.thread_id, run_id, prior_messages, long_term_keys, prior_values)
        try:
            result = graph.invoke(state, config=config)
        except Exception as e:
            _log.error("chat.graph_failed", extra={"attrs": {"thread_id": req.thread_id, "error": str(e)}}, exc_info=True)
            try:
                span.record_exception(e)
            except Exception:
                pass
            return {"response": "Sorry, something went wrong."}

        await store.save_messages(
            req.thread_id,
            result.get("messages", []),
            result.get("short_term_memory", []),
        )

        response = result.get("final_output") or result.get("error") or "Sorry, something went wrong."
        view = result.get("view")
        span.set_outputs({
            "response_length": len(response),
            "is_complete": bool(result.get("is_complete")),
            "intent": result.get("intent"),
            "location": result.get("location"),
            "view_type": (view or {}).get("type"),
        })
        return {"response": response, "view": view}


@router.post("/chat/trace")
async def chat_trace(req: ChatRequest):
    """Run the same graph as /chat but capture every node's update as a step.

    Powers the explanation UI at /trace.html — returns a structured trace the
    frontend animates step-by-step so users can see Planner → Orchestrator →
    Gate → Synthesizer execute.
    """
    graph = get_graph()
    if os.getenv("DEBUG_BREAK"):
        breakpoint()  # opt-in: only fires when DEBUG_BREAK is set (see .vscode/launch.json)
    run_id = uuid.uuid4().hex
    config = get_thread_config(req.thread_id + ":trace:" + run_id)  # isolated graph checkpoint per run

    # Load prior session memory + facts for this thread so consecutive traces on the
    # same thread_id build on each other — mirrors /chat. The trace UI keeps a stable
    # thread_id in localStorage, so reloads stay in the same session.
    store = get_mongo_store()
    prior_messages = await store.get_messages(req.thread_id)
    facts = await store.get_facts(req.thread_id)
    long_term_keys = [f"{k}: {v}" for k, v in facts.items()]

    state = _base_state(req.message, req.thread_id, run_id, prior_messages, long_term_keys)

    steps: list[dict] = []
    cumulative: dict = {}
    t0 = time.perf_counter()
    try:
        for chunk in graph.stream(state, config=config, stream_mode="updates"):
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                cumulative.update(update)
                slim = _slim_update(update)
                # db_ops lingers in state (every node returns full state via patch),
                # so it reappears on nodes that didn't produce it. Keep it only on the
                # nodes that actually touch a store.
                if node not in _DB_OP_PRODUCERS:
                    slim.pop("db_ops", None)
                steps.append({
                    "node": node,
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    "update": slim,
                })
    except Exception as e:
        _log.error("chat_trace.failed", extra={"attrs": {"error": str(e)}}, exc_info=True)
        return {"error": str(e), "steps": steps}

    # Persist this turn's session memory so the next trace on this thread sees it.
    # Reconstruct the conversation from known inputs + the final answer rather than
    # cumulative["messages"] — under stream_mode="updates" cumulative only holds the
    # last node's messages delta (the lone AIMessage), so it would drop the user turn.
    # Best-effort: the trace must still return even if the write fails.
    final_answer = cumulative.get("final_output") or cumulative.get("error") or ""
    turn = [HumanMessage(content=req.message)]
    if final_answer:
        turn.append(AIMessage(content=final_answer))
    try:
        await store.save_messages(
            req.thread_id,
            prior_messages + turn,
            cumulative.get("short_term_memory", []),
        )
    except Exception:
        _log.warning("chat_trace.save_messages_failed", extra={"attrs": {"thread_id": req.thread_id}})

    # Final response: the run is done, so clear this run's Redis blackboard mirror
    # (best-effort; the 1h TTL is the fallback). Session memory + the permanent
    # stores are untouched. Surfaced as a synthetic step so the trace shows cleanup.
    #
    # Only the executor writes the blackboard, so a run that never reached it (e.g.
    # the input guard blocked the prompt) has nothing in Redis under bb:thread:run:*.
    # Issuing a DEL then would be a no-op AND the card would falsely claim a clear,
    # so skip the call and report honestly instead.
    redis = get_redis_store()
    wrote_blackboard = bool(cumulative.get("blackboard"))
    if wrote_blackboard:
        try:
            await redis.delete_run(req.thread_id, run_id)
        except Exception:
            pass
        final_op = {
            "store": "redis",
            "op": "delete",
            "node": "final",
            "detail": "blackboard cleared (1h TTL is the fallback)",
            "code": f"DEL bb:{req.thread_id}:{run_id}:*",
            "enabled": redis.enabled,
        }
    else:
        final_op = {
            "store": "redis",
            "op": "delete",
            "node": "final",
            "detail": "no blackboard written this run — nothing to clear (no-op)",
            "code": f"DEL bb:{req.thread_id}:{run_id}:*  → 0 keys",
            "enabled": redis.enabled,
        }
    steps.append({
        "node": "final",
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "update": {"db_ops": [final_op]},
    })

    final_text = cumulative.get("final_output") or cumulative.get("error") or ""
    return {
        "user_input": req.message,
        "thread_id": req.thread_id,
        "run_id": run_id,
        "steps": steps,
        # What this run loaded from session memory BEFORE the graph ran, so the
        # trace UI can show prior turns carrying forward instead of starting cold.
        "session_loaded": {
            # Count prior exchanges (user turns), not raw messages — one
            # human+assistant turn reads as a single "prior message".
            "turns": sum(1 for m in prior_messages if isinstance(m, HumanMessage)),
            "preview": (str(getattr(prior_messages[-1], "content", "")) if prior_messages else "")[:80],
            "facts": long_term_keys,
        },
        "final": {
            "response": final_text,
            "view": cumulative.get("view"),
            "partial": bool(cumulative.get("partial")),
        },
    }


def _slim_update(update: dict) -> dict:
    """Strip noisy / heavy fields from a node update so the trace stays readable."""
    drop = {"messages", "long_term_memory_keys", "short_term_memory", "tool_calls", "tool_results"}
    out = {}
    for k, v in update.items():
        if k in drop:
            continue
        if isinstance(v, str) and len(v) > 2000:
            out[k] = v[:2000] + "...[truncated]"
        else:
            out[k] = v
    return out
