"""Unit tests for A2AClient response parsing (Task↔Message unwrap contract).

These prove the Executor/``call_peer`` contract survives the 1.2 move: a Task
result is unwrapped to a Message, and a failed Task becomes an A2AError exactly
as an agent-execution error did under 0.2.5.
"""
from __future__ import annotations

import pytest

from genie.a2a.client import A2AClient, A2AError
from genie.a2a.types import (
    Message,
    Task,
    TaskState,
    TaskStatus,
    JsonRpcResponse,
    get_text,
    text_part,
)


def _rpc(result) -> dict:
    return JsonRpcResponse(id="1", result=result.model_dump(mode="json")).model_dump(mode="json")


def test_parse_plain_message_result():
    msg = Message(role="agent", messageId="m1", parts=[text_part("hello")])
    out = A2AClient._parse_response(_rpc(msg))
    assert get_text(out) == "hello"


def test_parse_completed_task_unwraps_to_message():
    msg = Message(role="agent", messageId="m1", parts=[text_part("done")])
    task = Task(id="t1", status=TaskStatus(state=TaskState.completed, message=msg))
    out = A2AClient._parse_response(_rpc(task))
    assert get_text(out) == "done"


def test_parse_failed_task_raises_a2a_error():
    msg = Message(role="agent", messageId="m1", parts=[text_part("boom")])
    task = Task(id="t1", status=TaskStatus(state=TaskState.failed, message=msg))
    with pytest.raises(A2AError) as ei:
        A2AClient._parse_response(_rpc(task))
    assert "boom" in str(ei.value)


def test_parse_jsonrpc_error_raises():
    data = {"jsonrpc": "2.0", "id": "1", "error": {"code": -32601, "message": "nope"}}
    with pytest.raises(A2AError):
        A2AClient._parse_response(data)
