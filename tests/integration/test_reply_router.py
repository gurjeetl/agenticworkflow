"""Integration tests for the ReplyRouter's group-resume contract (no Kafka).

A run resumes exactly once — when every record of its dispatch group carries a
result. Covers: single-record groups, N-record fan-out groups, timeout beating
a late reply, step.cancelled converting dead-lettered records (the Supervisor's
DLQ hop), and the user-facing cancel_run.
"""
from __future__ import annotations

import asyncio
import json

from genie.interface.reply_router import ReplyRouter
from genie.messaging.awaiting import PENDING_STATUSES, STATUS_DEAD_LETTERED, STATUS_WAITING
from genie.messaging.broker import BusMessage, FakeBroker
from genie.messaging.dedup import Dedup
from genie.messaging.envelope import (
    HDR_CORRELATION_ID,
    HDR_ERROR,
    HDR_KIND,
    KIND_REPLY,
    KIND_STEP_CANCELLED,
    reply_topic,
)


class StubRedis:
    def __init__(self) -> None:
        self.keys: dict[str, str] = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.keys:
            return None
        self.keys[key] = value
        return True


class FakeAwaitingStore:
    """Dict-backed mirror of the AwaitingStore API surface the router uses."""

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    def add(self, cid, group_id, status=STATUS_WAITING, **kw):
        self.records[cid] = {"_id": cid, "group_id": group_id, "status": status,
                             "result": None, "thread_id": "thr-1", "run_id": "run-1", **kw}

    def get(self, cid):
        return self.records.get(cid)

    def resolve(self, cid, status="resolved", *, result=None, allowed_from=(STATUS_WAITING,)):
        rec = self.records.get(cid)
        if rec is None or rec["status"] not in allowed_from:
            return False
        rec["status"], rec["result"] = status, result
        return True

    def pending_in_group(self, group_id):
        return sum(1 for r in self.records.values()
                   if r["group_id"] == group_id and r["status"] in PENDING_STATUSES)

    def group_results(self, group_id):
        return {r.get("task_id", cid): r["result"]
                for cid, r in self.records.items()
                if r["group_id"] == group_id and r["result"] is not None}

    def waiting_for_run(self, thread_id, run_id):
        return [r for r in self.records.values()
                if r["thread_id"] == thread_id and r["run_id"] == run_id
                and r["status"] in PENDING_STATUSES]

    def expired(self, now=None):
        return [r for r in self.records.values() if r["status"] == STATUS_WAITING and r.get("_expired")]


def _router(store):
    resumed: list[tuple[dict, dict]] = []

    async def record_resume(rec, results):
        resumed.append((rec, results))

    router = ReplyRouter(
        broker=FakeBroker(),
        awaiting=store,
        dedup=Dedup(ttl_seconds=60, client=StubRedis()),
        resume_fn=record_resume,
    )
    return router, resumed


def _reply_bm(cid: str, text: str = "ok") -> BusMessage:
    return BusMessage(topic=reply_topic(), key="thr-1",
                      value=json.dumps({"kind": "task", "id": cid, "note": text}).encode(),
                      headers={HDR_KIND: KIND_REPLY, HDR_CORRELATION_ID: cid})


async def test_single_record_group_resumes_exactly_once():
    store = FakeAwaitingStore()
    store.add("cid-1", group_id="g1", task_id="t1", agent_id="a")
    router, resumed = _router(store)

    await router.handle_reply(_reply_bm("cid-1"))
    await router.handle_reply(_reply_bm("cid-1"))  # duplicate/late reply
    await asyncio.gather(*router._tasks)

    assert len(resumed) == 1
    assert resumed[0][1]["t1"]["task"]["id"] == "cid-1"
    assert store.records["cid-1"]["status"] == "resolved"


async def test_fanout_group_resumes_only_when_all_records_resolved():
    store = FakeAwaitingStore()
    store.add("cid-a", group_id="g2", task_id="t1", agent_id="a")
    store.add("cid-b", group_id="g2", task_id="t2", agent_id="b")
    router, resumed = _router(store)

    await router.handle_reply(_reply_bm("cid-a"))
    await asyncio.gather(*router._tasks)
    assert resumed == [], "half-resolved group must NOT resume"

    await router.handle_reply(_reply_bm("cid-b"))
    await asyncio.gather(*router._tasks)
    assert len(resumed) == 1
    assert set(resumed[0][1]) == {"t1", "t2"}  # combined payload for the Executor


async def test_expired_wait_resumes_with_deadline_error_and_beats_late_reply():
    store = FakeAwaitingStore()
    store.add("cid-2", group_id="g3", task_id="t1", agent_id="slowagent", _expired=True)
    router, resumed = _router(store)

    await router.expire(store.records["cid-2"])
    await router.handle_reply(_reply_bm("cid-2"))  # the real reply arrives too late
    await asyncio.gather(*router._tasks)

    assert len(resumed) == 1
    assert "deadline exceeded" in resumed[0][1]["t1"]["error"]
    assert store.records["cid-2"]["status"] == "timeout"


async def test_step_cancelled_converts_dead_lettered_record():
    store = FakeAwaitingStore()
    store.add("cid-3", group_id="g4", task_id="t1", agent_id="a", status=STATUS_DEAD_LETTERED)
    router, resumed = _router(store)

    bm = BusMessage(topic=reply_topic(), key="thr-1", value=b"{}", headers={
        HDR_KIND: KIND_STEP_CANCELLED, HDR_CORRELATION_ID: "cid-3",
        HDR_ERROR: "no reply before deadline — extensions and retries exhausted",
    })
    await router.handle_reply(bm)
    await asyncio.gather(*router._tasks)

    assert len(resumed) == 1
    assert "step cancelled" in resumed[0][1]["t1"]["error"]
    assert store.records["cid-3"]["status"] == "cancelled"


async def test_cancel_run_unblocks_every_pending_wait():
    store = FakeAwaitingStore()
    store.add("cid-x", group_id="g5", task_id="t1", agent_id="a")
    store.add("cid-y", group_id="g5", task_id="t2", agent_id="b")
    router, resumed = _router(store)

    count = await router.cancel_run("thr-1", "run-1")
    await asyncio.gather(*router._tasks)

    assert count == 2
    assert len(resumed) == 1  # one group → one resume with both cancellations
    assert all("cancelled by user request" in r["error"] for r in resumed[0][1].values())


async def test_unknown_cid_is_dropped_after_parking(monkeypatch):
    import genie.interface.reply_router as rr_mod

    monkeypatch.setattr(rr_mod, "_LOOKUP_RETRIES", 1)
    monkeypatch.setattr(rr_mod, "_LOOKUP_DELAY_S", 0.01)
    router, resumed = _router(FakeAwaitingStore())

    await router.handle_reply(_reply_bm("cid-nope"))
    await asyncio.gather(*router._tasks)
    assert resumed == []
