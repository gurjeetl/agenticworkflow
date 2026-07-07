"""Unit tests for the A2A wire types (a2a-sdk adapter): card version + Task lifecycle."""
from __future__ import annotations

import pytest

from genie.a2a.types import (
    PROTOCOL_VERSION,
    AgentCapabilities,
    AgentCard,
    Artifact,
    Message,
    Role,
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


def _card(**over):
    base = dict(
        name="x", description="d", url="http://h/a2a", version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text"], default_output_modes=["text"], skills=[],
    )
    base.update(over)
    return AgentCard(**base)


def test_agent_card_default_protocol_version_is_sdk_native():
    card = _card()
    assert card.protocol_version == "0.3.0" == PROTOCOL_VERSION


def test_task_round_trips_through_camelcase_json():
    msg = Message(role=Role.agent, message_id="m1", parts=[text_part("hi")])
    task = Task(id="t1", context_id="c1", status=TaskStatus(state=TaskState.completed, message=msg))
    wire = task.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert wire["kind"] == "task" and wire["contextId"] == "c1"
    restored = Task.model_validate(wire)
    assert restored.status.state is TaskState.completed
    assert get_text(restored.status.message) == "hi"


def test_status_and_artifact_update_events_validate():
    ev = TaskStatusUpdateEvent(task_id="t1", context_id="c1", status=TaskStatus(state=TaskState.working), final=False)
    assert ev.kind == "status-update" and ev.final is False
    art = Artifact(artifact_id="a1", parts=[data_part({"view": {"k": 1}})])
    aev = TaskArtifactUpdateEvent(task_id="t1", context_id="c1", artifact=art, last_chunk=True)
    assert aev.kind == "artifact-update" and aev.last_chunk is True


def test_get_text_and_get_data_read_part_root():
    from genie.a2a.types import get_data
    msg = Message(role=Role.agent, message_id="m1", parts=[text_part("answer"), data_part({"view": {"n": 2}})])
    assert get_text(msg) == "answer"
    assert get_data(msg) == {"view": {"n": 2}}


def test_task_final_message_prefers_status_message():
    msg = Message(role=Role.agent, message_id="m1", parts=[text_part("answer")])
    task = Task(id="t1", context_id="c1", status=TaskStatus(state=TaskState.completed, message=msg))
    assert get_text(task_final_message(task)) == "answer"


def test_task_final_message_falls_back_to_artifacts():
    art = Artifact(artifact_id="a1", parts=[text_part("from-artifact")])
    task = Task(id="t1", context_id="c1", status=TaskStatus(state=TaskState.completed), artifacts=[art])
    assert get_text(task_final_message(task)) == "from-artifact"


def test_task_final_message_raises_when_empty():
    task = Task(id="t1", context_id="c1", status=TaskStatus(state=TaskState.completed))
    with pytest.raises(ValueError):
        task_final_message(task)
