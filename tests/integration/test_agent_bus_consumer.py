"""Integration tests for the agent-side bus consumer (FakeBroker — no Kafka/Docker).

Drives ``_handle_inbox_message`` exactly as the inbox consumer loop does:
a valid request produces a terminal-Task reply on the reply topic; a malformed
payload is a poison pill that goes straight to the DLQ with no reply; a
redelivered (cid, attempt) is dropped by dedup.
"""
from __future__ import annotations

import json

from genie.a2a.types import Message, Role, Task, TaskState, data_part
from genie.agents.server import _handle_inbox_message
from genie.agents.task_store import TaskStore
from genie.messaging.broker import BusMessage, FakeBroker
from genie.messaging.dedup import Dedup
from genie.messaging.envelope import (
    HDR_ATTEMPT,
    HDR_CORRELATION_ID,
    HDR_ERROR,
    HDR_KIND,
    HDR_REPLY_TO,
    HDR_THREAD_ID,
    KIND_DEAD_LETTER,
    KIND_REPLY,
    KIND_REQUEST,
    dlq_topic,
    reply_topic,
)
from genie.registry.agent_meta import AgentMeta, FieldSpec
from tests.integration.test_agent_server import StubAgent


class StubRedis:
    def __init__(self) -> None:
        self.keys: dict[str, str] = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.keys:
            return None
        self.keys[key] = value
        return True


META = AgentMeta(
    agent_id="stub",
    capability_tags=["test"],
    description="stub",
    transport="kafka",
    input_schema={"location": FieldSpec(type="string", required=True)},
)


def _request_bm(cid: str = "cid-1", attempt: str = "1", *, value: bytes | None = None) -> BusMessage:
    msg = Message(
        role=Role.user,
        message_id="m1",
        parts=[data_part({"args": {"location": "Paris"}})],
        metadata={"agent_id": "stub", "task_id": "t1", "thread_id": "thr-1", "run_id": "run-1"},
    )
    return BusMessage(
        topic="genie.agents.stub.inbox",
        key="thr-1",
        value=value if value is not None else msg.model_dump_json(by_alias=True, exclude_none=True).encode(),
        headers={
            HDR_KIND: KIND_REQUEST,
            HDR_CORRELATION_ID: cid,
            HDR_ATTEMPT: attempt,
            HDR_REPLY_TO: reply_topic(),
            HDR_THREAD_ID: "thr-1",
        },
    )


async def test_valid_request_produces_completed_task_reply():
    broker, tasks, dedup = FakeBroker(), TaskStore(), Dedup(ttl_seconds=60, client=StubRedis())
    await _handle_inbox_message(broker, StubAgent(), META, tasks, dedup, _request_bm())

    replies = broker.log.get(reply_topic(), [])
    assert len(replies) == 1
    reply = replies[0]
    assert reply.headers[HDR_KIND] == KIND_REPLY
    assert reply.headers[HDR_CORRELATION_ID] == "cid-1"
    task = Task.model_validate(json.loads(reply.value))
    assert task.status.state is TaskState.completed
    assert "weather in Paris" in (task.status.message.parts[0].root.text or "")
    assert tasks.get(task.id) is not None  # tasks/get keeps working for bus tasks
    assert not broker.log.get(dlq_topic())


async def test_poison_pill_goes_to_dlq_without_reply():
    broker, tasks, dedup = FakeBroker(), TaskStore(), Dedup(ttl_seconds=60, client=StubRedis())
    await _handle_inbox_message(broker, StubAgent(), META, tasks, dedup, _request_bm(value=b"{not json"))

    dead = broker.log.get(dlq_topic(), [])
    assert len(dead) == 1
    assert dead[0].headers[HDR_KIND] == KIND_DEAD_LETTER
    assert "schema_validation_failed" in dead[0].headers[HDR_ERROR]
    assert dead[0].value == b"{not json"  # original payload preserved for replay
    assert not broker.log.get(reply_topic())  # no reply — sweep/Supervisor unblocks the caller


async def test_redelivered_attempt_is_deduped_but_retry_passes():
    broker, tasks, dedup = FakeBroker(), TaskStore(), Dedup(ttl_seconds=60, client=StubRedis())
    agent = StubAgent()
    await _handle_inbox_message(broker, agent, META, tasks, dedup, _request_bm(attempt="1"))
    await _handle_inbox_message(broker, agent, META, tasks, dedup, _request_bm(attempt="1"))  # redelivery
    assert len(broker.log.get(reply_topic(), [])) == 1  # duplicate dropped, no double-run

    await _handle_inbox_message(broker, agent, META, tasks, dedup, _request_bm(attempt="2"))  # retry
    assert len(broker.log.get(reply_topic(), [])) == 2
