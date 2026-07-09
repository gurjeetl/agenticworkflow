"""Integration tests for the Executor's durable bus suspend/resume (no Kafka).

Builds a minimal executor-only StateGraph with a MemorySaver checkpoint and
fakes at the seams (registry, send_via_bus): a kafka-transport wave must
produce every request with deterministic correlation ids, suspend ONCE for the
whole group, and apply the combined resume payload ``{task_id: result}`` to the
blackboard. Also covers the transport="both" fast path (direct call fails →
same wave re-enters over the bus).
"""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

import genie.platform.config as cfg
from genie.a2a.client import A2AError
from genie.a2a.types import Message, Role, Task, TaskState, TaskStatus, text_part
from genie.application.graph import route_after_executor
from genie.application.nodes.executor import Executor
from genie.application.state import AgentState
from genie.messaging.envelope import correlation_id
from genie.registry.agent_meta import AgentMeta

CONFIG = {"configurable": {"thread_id": "thr-1"}}


class FakeRegistry:
    def __init__(self, meta: AgentMeta):
        self.meta = meta

    def get(self, agent_id):
        return self.meta

    def invalidate(self):
        pass


def _graph_env(monkeypatch, transport: str, sync_send=None):
    """Executor graph with faked bus/sync seams; returns (graph, produced)."""
    meta = AgentMeta(agent_id="busagent", capability_tags=["test"], transport=transport, endpoint="http://x")
    ex = Executor()
    ex._registry = FakeRegistry(meta)
    produced: list[dict] = []

    async def fake_send_via_bus(agent_id, args, context, *, attempt=1, deadline_ms=None, group_id=None, broker=None):
        produced.append({"agent_id": agent_id, "args": args, "cid": context["correlation_id"],
                         "attempt": attempt, "group": group_id})
        return context["correlation_id"]

    monkeypatch.setattr(ex._a2a, "send_via_bus", fake_send_via_bus)
    if sync_send is not None:
        monkeypatch.setattr(ex._a2a, "send", sync_send)

    g = StateGraph(AgentState)
    g.add_node("executor", ex.run)
    g.add_edge(START, "executor")
    g.add_conditional_edges("executor", route_after_executor, {"executor": "executor", "gate": END})
    return g.compile(checkpointer=MemorySaver()), produced


@pytest.fixture
def kafka_on():
    base = cfg.get_settings()
    cfg.override_settings(base.model_copy(update={"kafka_enabled": True}))
    yield
    cfg.override_settings(base)


# NOTE: every test uses a UNIQUE run_id. The blackboard read-through resolves
# entries from the Redis mirror by (tenant, thread, run, task) — in production
# run ids are uuid4-unique, but reusing one across tests would let an earlier
# test's mirrored result legitimately satisfy a later test's task.
def _state(run_id, subtasks=None, waves=None) -> dict:
    subtasks = subtasks or [{"id": "t1", "agent_id": "busagent", "args": {"location": "Paris"}}]
    return {
        "user_input": "q",
        "thread_id": "thr-1",
        "run_id": run_id,
        "plan": {"subtasks": subtasks},
        "waves": waves or [[t["id"] for t in subtasks]],
        "wave_cursor": 0,
        "blackboard": {},
        "messages": [],
    }


def _task_payload(text: str, task_id: str = "t1", state: TaskState = TaskState.completed) -> dict:
    msg = Message(role=Role.agent, message_id="m1", parts=[text_part(text)])
    task = Task(id=task_id, context_id="thr-1", status=TaskStatus(state=state, message=msg))
    return {"task": task.model_dump(mode="json", by_alias=True, exclude_none=True)}


def test_bus_wave_suspends_once_with_deterministic_cids(kafka_on, monkeypatch):
    graph, produced = _graph_env(monkeypatch, "kafka")
    result = graph.invoke(_state("run-s1"), CONFIG)

    assert result.get("__interrupt__"), "bus task must suspend the run"
    assert produced[0]["cid"] == correlation_id("run-s1", "t1", 1)
    assert produced[0]["group"] == "run-s1:0"  # deterministic group = run:wave


def test_resume_with_group_results_fills_blackboard(kafka_on, monkeypatch):
    graph, produced = _graph_env(monkeypatch, "kafka")
    graph.invoke(_state("run-s2"), CONFIG)
    result = graph.invoke(Command(resume={"t1": _task_payload("bus says hi")}), CONFIG)

    assert not result.get("__interrupt__")
    assert result["blackboard"]["t1"]["text"] == "bus says hi"
    assert result["wave_cursor"] == 1
    # Node re-execution re-produced the request — with the SAME cid, so the
    # agent-side dedup drops it instead of double-running the agent.
    assert len(produced) >= 2 and len({p["cid"] for p in produced}) == 1


def test_parallel_fanout_two_bus_tasks_one_interrupt(kafka_on, monkeypatch):
    subtasks = [
        {"id": "t1", "agent_id": "busagent", "args": {"a": 1}},
        {"id": "t2", "agent_id": "busagent", "args": {"a": 2}},
    ]
    graph, produced = _graph_env(monkeypatch, "kafka")
    result = graph.invoke(_state("run-fan", subtasks=subtasks, waves=[["t1", "t2"]]), CONFIG)

    assert result.get("__interrupt__")
    assert len(result["__interrupt__"]) == 1, "whole wave suspends on ONE interrupt"
    assert {p["cid"] for p in produced} == {correlation_id("run-fan", "t1", 1), correlation_id("run-fan", "t2", 1)}
    assert {p["group"] for p in produced} == {"run-fan:0"}  # same fan-out group

    result = graph.invoke(
        Command(resume={"t1": _task_payload("one", "t1"), "t2": _task_payload("two", "t2")}), CONFIG
    )
    assert result["blackboard"]["t1"]["text"] == "one"
    assert result["blackboard"]["t2"]["text"] == "two"
    assert result["wave_cursor"] == 1


def test_resume_with_error_and_failed_task_record_blackboard_errors(kafka_on, monkeypatch):
    graph, _ = _graph_env(monkeypatch, "kafka")
    graph.invoke(_state("run-err1"), CONFIG)
    result = graph.invoke(Command(resume={"t1": {"error": "deadline exceeded waiting for agent 'busagent'"}}), CONFIG)
    assert "deadline exceeded" in result["blackboard"]["t1"]["error"]

    graph, _ = _graph_env(monkeypatch, "kafka")
    graph.invoke(_state("run-err2"), CONFIG)
    result = graph.invoke(
        Command(resume={"t1": _task_payload("kaboom", state=TaskState.failed)}), CONFIG
    )
    assert "kaboom" in result["blackboard"]["t1"]["error"]


def test_fast_path_both_falls_back_to_bus_on_sync_failure(kafka_on, monkeypatch):
    async def failing_send(agent_id, args, context, *, sla_ms, http=None):
        raise A2AError("connection refused")

    graph, produced = _graph_env(monkeypatch, "both", sync_send=failing_send)
    result = graph.invoke(_state("run-fp"), CONFIG)

    # Direct call failed → the SAME wave re-entered with the task on the bus.
    assert result.get("__interrupt__")
    assert result.get("bus_fallback") == ["t1"]
    assert produced and produced[0]["cid"] == correlation_id("run-fp", "t1", 1)

    result = graph.invoke(Command(resume={"t1": _task_payload("bus rescue")}), CONFIG)
    assert result["blackboard"]["t1"]["text"] == "bus rescue"
    assert result["wave_cursor"] == 1
    assert not result.get("bus_fallback")
