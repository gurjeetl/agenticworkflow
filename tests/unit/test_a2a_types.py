"""Unit tests for the A2A v1.2 wire types (card version, Task lifecycle)."""
from __future__ import annotations

import pytest

from genie.a2a.types import (
    PROTOCOL_VERSION,
    AgentCard,
    Artifact,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    data_part,
    get_text,
    task_final_message,
    text_part,
)


def test_agent_card_defaults_to_protocol_1_2():
    card = AgentCard(name="x", url="http://host/a2a")
    assert card.protocolVersion == "1.2" == PROTOCOL_VERSION
    assert card.preferredTransport == "JSONRPC"


def test_task_round_trips_through_json():
    msg = Message(role="agent", messageId="m1", parts=[text_part("hi")])
    task = Task(id="t1", contextId="c1", status=TaskStatus(state=TaskState.completed, message=msg))
    restored = Task.model_validate(task.model_dump(mode="json"))
    assert restored.kind == "task"
    assert restored.status.state is TaskState.completed
    assert get_text(restored.status.message) == "hi"


def test_status_and_artifact_update_events_validate():
    status = TaskStatusUpdateEvent(taskId="t1", status=TaskStatus(state=TaskState.working))
    assert status.kind == "status-update" and status.final is False
    art = Artifact(artifactId="a1", parts=[data_part({"view": {"k": 1}})])
    ev = TaskArtifactUpdateEvent(taskId="t1", artifact=art, lastChunk=True)
    assert ev.kind == "artifact-update" and ev.lastChunk is True


def test_task_final_message_prefers_status_message():
    msg = Message(role="agent", messageId="m1", parts=[text_part("answer")])
    task = Task(id="t1", status=TaskStatus(state=TaskState.completed, message=msg))
    assert get_text(task_final_message(task)) == "answer"


def test_task_final_message_falls_back_to_artifacts():
    art = Artifact(artifactId="a1", parts=[text_part("from-artifact")])
    task = Task(id="t1", status=TaskStatus(state=TaskState.completed), artifacts=[art])
    assert get_text(task_final_message(task)) == "from-artifact"


def test_task_final_message_raises_when_empty():
    task = Task(id="t1", status=TaskStatus(state=TaskState.completed))
    with pytest.raises(ValueError):
        task_final_message(task)
