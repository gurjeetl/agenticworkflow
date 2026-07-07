"""Unit tests for A2AClient response parsing (Task↔Message unwrap contract).

These prove the Executor/``call_peer`` contract survives the a2a-sdk adoption: a
Task result is unwrapped to a Message, and a failed Task becomes an A2AError.
"""
from __future__ import annotations

import pytest

from genie.a2a.client import A2AClient, A2AError
from genie.a2a.types import (
    Message,
    Role,
    Task,
    TaskState,
    TaskStatus,
    get_text,
    text_part,
)


def _ok(payload) -> dict:
    return {"jsonrpc": "2.0", "id": "1", "result": payload.model_dump(mode="json", by_alias=True, exclude_none=True)}


def test_parse_plain_message_result():
    msg = Message(role=Role.agent, message_id="m1", parts=[text_part("hello")])
    out = A2AClient._parse_response(_ok(msg))
    assert get_text(out) == "hello"


def test_parse_completed_task_unwraps_to_message():
    msg = Message(role=Role.agent, message_id="m1", parts=[text_part("done")])
    task = Task(id="t1", context_id="c1", status=TaskStatus(state=TaskState.completed, message=msg))
    out = A2AClient._parse_response(_ok(task))
    assert get_text(out) == "done"


def test_parse_failed_task_raises_a2a_error():
    msg = Message(role=Role.agent, message_id="m1", parts=[text_part("boom")])
    task = Task(id="t1", context_id="c1", status=TaskStatus(state=TaskState.failed, message=msg))
    with pytest.raises(A2AError) as ei:
        A2AClient._parse_response(_ok(task))
    assert "boom" in str(ei.value)


def test_parse_jsonrpc_error_raises_with_code():
    data = {"jsonrpc": "2.0", "id": "1", "error": {"code": -32001, "message": "Task not found"}}
    with pytest.raises(A2AError) as ei:
        A2AClient._parse_response(data)
    assert ei.value.code == -32001
