from dotenv import load_dotenv
load_dotenv()

import uuid

import uvicorn
import mlflow
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from mlflow.entities import SpanType
from pydantic import BaseModel

from observability import init_mlflow, configure_logging, get_logger

configure_logging()
init_mlflow()

from graph.graph_builder import build_graph
from memory.memory import get_thread_config
from memory.mongo_store import get_mongo_store
from memory.postgres_store import get_postgres_store
from memory.redis_store import get_redis_store

_log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_mongo_store()
    await store.ensure_indexes()
    _log.info("mongodb.indexes_ensured")

    pg = get_postgres_store()
    await pg.ensure_pool()
    _log.info("postgres.ready", extra={"attrs": {"enabled": pg.enabled}})

    redis = get_redis_store()
    _log.info("redis.ready", extra={"attrs": {"enabled": redis.enabled}})

    yield

    store.close()
    await pg.close()
    await redis.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()


class ChatRequest(BaseModel):
    message: str
    thread_id: str


@app.post("/chat")
async def chat(req: ChatRequest):
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
        state = {
            "user_input": req.message,
            "current_task": "",
            "thread_id": req.thread_id,
            "run_id": run_id,
            "messages": prior_messages + [HumanMessage(content=req.message)],
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
            "plan": None,
            "agent_versions": {},
            "blackboard": {},
            "blackboard_snapshot": None,
            "replan_count": 0,
            "max_replans": 3,
            "replan_reason": None,
            "partial": False,
            "final_output": None,
            "view": None,
            "is_complete": False,
            "error": None,
        }
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


@app.post("/chat/trace")
async def chat_trace(req: ChatRequest):
    """Run the same graph as /chat but capture every node's update as a step.

    Powers the explanation UI at /trace.html — returns a structured trace the
    frontend animates step-by-step so users can see Planner → Orchestrator →
    Gate → Synthesizer execute.
    """
    import time
    run_id = uuid.uuid4().hex
    config = get_thread_config(req.thread_id + ":trace:" + run_id)  # isolated thread to avoid polluting chat history
    state = {
        "user_input": req.message,
        "current_task": "",
        "thread_id": req.thread_id,
        "run_id": run_id,
        "messages": [HumanMessage(content=req.message)],
        "agent_scratchpad": "",
        "iteration_count": 0,
        "max_iterations": 10,
        "tool_calls": [],
        "tool_results": [],
        "short_term_memory": [],
        "long_term_memory_keys": [],
        "active_agent": "",
        "next_action": "",
        "delegated_task": None,
        "location": None,
        "intent": None,
        "outage_id": None,
        "plan": None,
        "agent_versions": {},
        "blackboard": {},
        "blackboard_snapshot": None,
        "replan_count": 0,
        "max_replans": 3,
        "replan_reason": None,
        "partial": False,
        "final_output": None,
        "view": None,
        "is_complete": False,
        "error": None,
    }

    steps: list[dict] = []
    cumulative: dict = {}
    t0 = time.perf_counter()
    try:
        for chunk in graph.stream(state, config=config, stream_mode="updates"):
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                cumulative.update(update)
                steps.append({
                    "node": node,
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    "update": _slim_update(update),
                })
    except Exception as e:
        _log.error("chat_trace.failed", extra={"attrs": {"error": str(e)}}, exc_info=True)
        return {"error": str(e), "steps": steps}

    final_text = cumulative.get("final_output") or cumulative.get("error") or ""
    return {
        "user_input": req.message,
        "thread_id": req.thread_id,
        "run_id": run_id,
        "steps": steps,
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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/state/{thread_id}")
async def get_state(thread_id: str):
    config = get_thread_config(thread_id)
    snapshot = graph.get_state(config)
    return snapshot.values


@app.get("/registry")
async def registry_dump():
    """Expose registered agents so the trace UI can show the menu the Planner saw."""
    from registry import list_active
    return {
        "agents": [
            {
                "agent_id": m.agent_id,
                "version": m.version,
                "capability_tags": m.capability_tags,
                "description": m.description,
                "input_schema": {k: v.model_dump() for k, v in m.input_schema.items()},
                "output_schema": {k: v.model_dump() for k, v in m.output_schema.items()},
                "sla_ms": m.sla_ms,
                "transport": m.transport,
                "status": m.status,
            }
            for m, _cls in list_active()
        ]
    }


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
