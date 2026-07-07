"""Integration tests for the A2A agent harness (create_agent_app) on a2a-sdk types.

Uses a stub agent (no LLM/MCP) so the tests exercise only the A2A surface: card
discovery, message/send → Task, tasks/get, tasks/cancel, and message/stream (SSE).
Wire assertions use the SDK's camelCase JSON; error codes are the A2A standard ones.
TestClient is used without the lifespan context manager, so no Registry connection
is attempted — proving an agent is independently testable.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import genie.platform.config as cfg
from genie.agents.server import create_agent_app
from genie.a2a.types import (
    METHOD_MESSAGE_SEND,
    METHOD_MESSAGE_STREAM,
    METHOD_TASKS_CANCEL,
    METHOD_TASKS_GET,
)
from genie.registry.agent_meta import AgentMeta, FieldSpec


class StubAgent:
    """Minimal agent: echoes its location arg and returns a structured view."""

    def __init__(self) -> None:  # no LLM/MCP wiring
        pass

    def run(self, state: dict) -> dict:
        loc = state.get("location")
        return {**state, "final_output": f"weather in {loc}", "view": {"loc": loc}, "error": None}


class FailAgent:
    def __init__(self) -> None:
        pass

    def run(self, state: dict) -> dict:
        return {**state, "error": "kaboom", "final_output": None}


META = AgentMeta(
    agent_id="stub",
    capability_tags=["test"],
    description="stub",
    input_schema={"location": FieldSpec(type="string", required=True)},
)


def _send_body(method: str, args: dict, task_id: str | None = "task-1") -> dict:
    metadata = {"agent_id": "stub", "thread_id": "thread-1"}
    if task_id is not None:
        metadata["task_id"] = task_id
    return {
        "jsonrpc": "2.0",
        "id": "rpc-1",
        "method": method,
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "messageId": "m1",
                "parts": [{"kind": "data", "data": {"args": args}}],
                "metadata": metadata,
            }
        },
    }


@pytest.fixture
def client():
    base = cfg.get_settings()
    cfg.override_settings(base.model_copy(update={"agent_invoke_token": None}))
    app = create_agent_app(StubAgent, META, port=0)
    yield TestClient(app)
    cfg.override_settings(base)


def test_agent_card_endpoint_is_sdk_native(client):
    card = client.get("/.well-known/agent-card.json").json()
    assert card["protocolVersion"] == "0.3.0"
    assert card["capabilities"]["streaming"] is True
    assert "securitySchemes" not in card  # token-free agent → omitted, open /a2a


def test_message_send_returns_completed_task(client):
    result = client.post("/a2a", json=_send_body(METHOD_MESSAGE_SEND, {"location": "Paris"})).json()["result"]
    assert result["kind"] == "task"
    assert result["status"]["state"] == "completed"
    text = "".join(p.get("text", "") for p in result["status"]["message"]["parts"])
    assert text == "weather in Paris"


def test_tasks_get_returns_stored_task(client):
    client.post("/a2a", json=_send_body(METHOD_MESSAGE_SEND, {"location": "Paris"}, task_id="task-xyz"))
    result = client.post("/a2a", json={"jsonrpc": "2.0", "id": "2", "method": METHOD_TASKS_GET, "params": {"id": "task-xyz"}}).json()["result"]
    assert result["id"] == "task-xyz"
    assert result["status"]["state"] == "completed"


def test_tasks_get_unknown_id_returns_task_not_found(client):
    resp = client.post("/a2a", json={"jsonrpc": "2.0", "id": "2", "method": METHOD_TASKS_GET, "params": {"id": "nope"}})
    assert resp.json()["error"]["code"] == -32001  # TaskNotFoundError


def test_tasks_cancel_terminal_returns_not_cancelable(client):
    client.post("/a2a", json=_send_body(METHOD_MESSAGE_SEND, {"location": "Paris"}, task_id="task-c"))
    resp = client.post("/a2a", json={"jsonrpc": "2.0", "id": "3", "method": METHOD_TASKS_CANCEL, "params": {"id": "task-c"}})
    assert resp.json()["error"]["code"] == -32002  # TaskNotCancelableError (synchronous task already terminal)


def test_unknown_method_returns_method_not_found(client):
    resp = client.post("/a2a", json={"jsonrpc": "2.0", "id": "2", "method": "bogus/thing", "params": {}})
    assert resp.json()["error"]["code"] == -32601  # MethodNotFoundError


def test_message_stream_emits_lifecycle_events(client):
    resp = client.post("/a2a", json=_send_body(METHOD_MESSAGE_STREAM, {"location": "Rome"}))
    assert resp.status_code == 200
    frames = [json.loads(line[len("data:"):].strip()) for line in resp.text.splitlines() if line.startswith("data:")]
    kinds = [f["result"]["kind"] for f in frames]
    assert kinds[0] == "task"  # initial submitted task
    assert "status-update" in kinds and "artifact-update" in kinds
    last = frames[-1]["result"]
    assert last["kind"] == "status-update" and last["final"] is True
    assert last["status"]["state"] == "completed"


def test_stream_frames_have_string_context_id_without_caller_context(client):
    body = _send_body(METHOD_MESSAGE_STREAM, {"location": "Paris"}, task_id=None)
    body["params"]["message"]["metadata"].pop("thread_id")
    frames = [json.loads(line[len("data:"):].strip()) for line in client.post("/a2a", json=body).text.splitlines() if line.startswith("data:")]
    ctx_ids = {f["result"].get("contextId") for f in frames}
    assert ctx_ids and all(isinstance(c, str) and c for c in ctx_ids)
    assert len(ctx_ids) == 1  # same contextId across the whole stream
    assert all("error" not in f for f in frames)  # exclude_none: no null error on success frames


def test_message_send_without_context_id_mints_string(client):
    body = _send_body(METHOD_MESSAGE_SEND, {"location": "Paris"}, task_id=None)
    body["params"]["message"]["metadata"].pop("thread_id")
    result = client.post("/a2a", json=body).json()["result"]
    assert isinstance(result["contextId"], str) and result["contextId"]


def test_streaming_disabled_agent_hides_endpoint():
    base = cfg.get_settings()
    cfg.override_settings(base.model_copy(update={"agent_invoke_token": None}))
    try:
        no_stream_meta = META.model_copy(update={"supports_streaming": False})
        c = TestClient(create_agent_app(StubAgent, no_stream_meta, port=0))
        assert c.get("/.well-known/agent-card.json").json()["capabilities"]["streaming"] is False
        resp = c.post("/a2a", json=_send_body(METHOD_MESSAGE_STREAM, {"location": "X"}))
        assert resp.json()["error"]["code"] == -32004  # UnsupportedOperationError
    finally:
        cfg.override_settings(base)


def test_message_send_failed_agent_returns_failed_task():
    base = cfg.get_settings()
    cfg.override_settings(base.model_copy(update={"agent_invoke_token": None}))
    try:
        c = TestClient(create_agent_app(FailAgent, META, port=0))
        result = c.post("/a2a", json=_send_body(METHOD_MESSAGE_SEND, {"location": "X"})).json()["result"]
        assert result["kind"] == "task"
        assert result["status"]["state"] == "failed"
    finally:
        cfg.override_settings(base)
