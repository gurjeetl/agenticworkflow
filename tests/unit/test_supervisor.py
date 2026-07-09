"""Unit tests for the A2A Supervisor's failure ladder (all collaborators faked).

extend (heartbeat healthy, budget left) → retry (attempt+1, new deterministic
cid, same group) → dead-letter (payload preserved) → step.cancelled unblocking
the waiting run. Mirrors the walkthrough diagram's failure path exactly.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from genie.messaging.awaiting import PENDING_STATUSES, STATUS_DEAD_LETTERED, STATUS_WAITING
from genie.messaging.broker import BusMessage, FakeBroker
from genie.messaging.envelope import (
    HDR_ATTEMPT,
    HDR_CORRELATION_ID,
    HDR_ERROR,
    HDR_KIND,
    KIND_DEAD_LETTER,
    KIND_REQUEST,
    KIND_STEP_CANCELLED,
    correlation_id,
    dlq_topic,
    reply_topic,
)
from genie.messaging.supervisor import Supervisor


class FakeAwaitingStore:
    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    def add(self, cid, **kw):
        self.records[cid] = {"_id": cid, "status": STATUS_WAITING, "extends_used": 0,
                             "attempt": 1, "group_id": "g1", "thread_id": "thr-1",
                             "run_id": "run-1", "task_id": "t1", "agent_id": "slowagent",
                             "inbox_topic": "genie.agents.slowagent.inbox",
                             "request": json.dumps({"kind": "message", "metadata": {"correlation_id": cid}}),
                             **kw}
        return self.records[cid]

    def get(self, cid):
        return self.records.get(cid)

    def extend(self, cid, new_deadline):
        rec = self.records.get(cid)
        if rec is None or rec["status"] != STATUS_WAITING:
            return False
        rec["deadline"] = new_deadline
        rec["extends_used"] += 1
        return True

    def mark_retried(self, cid):
        rec = self.records.get(cid)
        if rec is None or rec["status"] != STATUS_WAITING:
            return False
        rec["status"] = "retried"
        return True

    def mark_dead_lettered(self, cid):
        rec = self.records.get(cid)
        if rec is None or rec["status"] != STATUS_WAITING:
            return False
        rec["status"] = STATUS_DEAD_LETTERED
        return True

    def put(self, cid, **kw):
        self.records.setdefault(cid, {"_id": cid, "status": STATUS_WAITING, "result": None,
                                      "extends_used": 0, **kw})

    def expired(self, now=None):
        return []

    def pending_in_group(self, group_id):
        return sum(1 for r in self.records.values()
                   if r["group_id"] == group_id and r["status"] in PENDING_STATUSES)


class FakeRegistry:
    def __init__(self, healthy: bool):
        self.healthy = healthy

    def get(self, agent_id):
        return object() if self.healthy else None


def _supervisor(store, healthy=True):
    broker = FakeBroker()
    return Supervisor(broker=broker, awaiting=store, registry=FakeRegistry(healthy)), broker


async def test_expired_with_healthy_agent_gets_extension_not_retry():
    store = FakeAwaitingStore()
    rec = store.add("cid-1", deadline=datetime.now(timezone.utc) - timedelta(seconds=1))
    sup, broker = _supervisor(store, healthy=True)

    await sup.handle_expired(rec)

    assert store.records["cid-1"]["extends_used"] == 1  # "granted — 1 of 3"
    assert store.records["cid-1"]["status"] == STATUS_WAITING
    assert not broker.log  # no produce: B is slow, not dead


async def test_extensions_exhausted_retries_with_new_cid_same_group():
    store = FakeAwaitingStore()
    rec = store.add("cid-old", extends_used=3)  # budget spent
    sup, broker = _supervisor(store, healthy=True)

    await sup.handle_expired(rec)

    assert store.records["cid-old"]["status"] == "retried"
    new_cid = correlation_id("run-1", "t1", 2)
    assert new_cid in store.records and store.records[new_cid]["attempt"] == 2
    assert store.records[new_cid]["group_id"] == "g1"  # group stays open, resume waits for attempt 2

    sent = broker.log["genie.agents.slowagent.inbox"][0]
    assert sent.headers[HDR_KIND] == KIND_REQUEST
    assert sent.headers[HDR_CORRELATION_ID] == new_cid
    assert sent.headers[HDR_ATTEMPT] == "2"
    assert json.loads(sent.value)["metadata"]["correlation_id"] == new_cid  # body kept consistent


async def test_unhealthy_agent_skips_extension_and_goes_down_the_ladder():
    store = FakeAwaitingStore()
    rec = store.add("cid-2", extends_used=0)  # budget available, but agent is dead
    sup, broker = _supervisor(store, healthy=False)

    await sup.handle_expired(rec)

    assert store.records["cid-2"]["status"] == "retried"  # no pointless extension


async def test_retries_exhausted_dead_letters_with_payload_preserved():
    store = FakeAwaitingStore()
    rec = store.add("cid-3", extends_used=3, attempt=2)  # bus_max_attempts default = 2
    sup, broker = _supervisor(store, healthy=False)

    await sup.handle_expired(rec)

    assert store.records["cid-3"]["status"] == STATUS_DEAD_LETTERED
    dead = broker.log[dlq_topic()][0]
    assert dead.headers[HDR_KIND] == KIND_DEAD_LETTER
    assert dead.headers[HDR_CORRELATION_ID] == "cid-3"
    assert "exhausted" in dead.headers[HDR_ERROR]
    assert json.loads(dead.value)["metadata"]["correlation_id"] == "cid-3"  # replayable


async def test_dlq_consumer_emits_step_cancelled_for_pending_waits_only():
    store = FakeAwaitingStore()
    store.add("cid-4", status=STATUS_DEAD_LETTERED)
    store.add("cid-5", status="resolved")  # already handled — must be skipped
    sup, broker = _supervisor(store)

    dead = BusMessage(topic=dlq_topic(), key="thr-1", value=b"{}", headers={
        HDR_KIND: KIND_DEAD_LETTER, HDR_CORRELATION_ID: "cid-4", HDR_ERROR: "poison",
    })
    await sup.handle_dead_letter(dead)
    await sup.handle_dead_letter(BusMessage(topic=dlq_topic(), key=None, value=b"{}",
                                            headers={HDR_CORRELATION_ID: "cid-5"}))

    cancels = broker.log.get(reply_topic(), [])
    assert len(cancels) == 1  # only the pending wait was unblocked
    assert cancels[0].headers[HDR_KIND] == KIND_STEP_CANCELLED
    assert cancels[0].headers[HDR_CORRELATION_ID] == "cid-4"
    assert cancels[0].headers[HDR_ERROR] == "poison"
