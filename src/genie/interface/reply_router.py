"""Gateway reply-router: resumes suspended runs from the A2A reply topic.

The counterpart of the Executor's ``interrupt()``. Every bus dispatch of one
Executor wave belongs to a **group**; each incoming record (real reply,
``step.cancelled`` control message, deadline timeout, user cancel) resolves its
``a2a_awaiting`` record with a result, and the run resumes exactly once — when
the group has no pending records left — with the combined payload
``{task_id: {"task": ...} | {"error": ...}}``.

Concurrency guards, in order:
1. per-record: atomic Mongo status transition + Redis ``dedup:reply:{cid}``
   claim — first of reply/timeout/cancel wins, late arrivals are ignored;
2. per-group: ``dedup:resume:{group_id}`` claim — exactly one resume even when
   two records resolve simultaneously on different instances.

The consumer loop never runs the graph inline — resumes happen in background
tasks so a slow synthesizer can't stall the Kafka consumer group into a
rebalance. The **deadline sweep** runs here only while the Phase 2 Supervisor is
disabled (``bus_supervisor_enabled=false``); with the Supervisor running, it
owns the richer extend → retry → dead-letter ladder and this sweep stands down.
"""
from __future__ import annotations

import asyncio
import json

from langgraph.types import Command

from genie.application.checkpointer import get_thread_config
from genie.application.graph import get_graph
from genie.memory.mongo_store import get_mongo_store
from genie.messaging import Dedup, get_awaiting_store, get_broker
from genie.messaging.awaiting import (
    STATUS_CANCELLED,
    STATUS_DEAD_LETTERED,
    STATUS_RESOLVED,
    STATUS_TIMEOUT,
    STATUS_WAITING,
)
from genie.messaging.broker import BusMessage
from genie.messaging.envelope import (
    HDR_CORRELATION_ID,
    HDR_ERROR,
    HDR_KIND,
    KIND_STEP_CANCELLED,
    reply_topic,
)
from genie.observability import get_logger
from genie.platform.config import get_settings

_log = get_logger(__name__)

# A reply can only beat its own awaiting record in pathological clock/broker
# situations (the record is written BEFORE the produce); park-and-retry briefly
# instead of dropping the reply.
_LOOKUP_RETRIES = 4
_LOOKUP_DELAY_S = 0.5
# The suspend may not be committed to the checkpoint yet when the group
# completes; poll for the pending interrupt before resuming.
_RESUME_POLLS = 40
_RESUME_POLL_DELAY_S = 0.25


class ReplyRouter:
    """Reply consumer + group resume + deadline sweep. Collaborators injectable for tests."""

    def __init__(self, broker=None, awaiting=None, dedup=None, resume_fn=None) -> None:
        """Default to the process singletons; tests pass fakes."""
        self._broker = broker or get_broker()
        self._store = awaiting or get_awaiting_store()
        self._dedup = dedup or Dedup()
        self._resume_fn = resume_fn or self._resume
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Launch the consumer loop (+ the deadline sweep unless the Supervisor owns it)."""
        self._tasks.append(asyncio.create_task(self._consume_loop()))
        if not get_settings().bus_supervisor_enabled:
            self._tasks.append(asyncio.create_task(self._sweep_loop()))
        _log.info("a2a.reply_router.started", extra={"attrs": {
            "topic": reply_topic(),
            "sweep": not get_settings().bus_supervisor_enabled,
        }})

    async def stop(self) -> None:
        """Cancel every loop and in-flight resume task."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    # ------------------------------------------------------------------
    async def _consume_loop(self) -> None:
        """Consume the shared reply topic forever; one handler call per record."""
        group = get_settings().bus_consumer_group
        async for bm in self._broker.consume([reply_topic()], group=group):
            try:
                await self.handle_reply(bm)
            except Exception as e:
                _log.error("a2a.reply_router.handle_failed", extra={"attrs": {"error": str(e)}}, exc_info=True)

    async def handle_reply(self, bm: BusMessage) -> None:
        """Resolve one reply/control record and resume its group if now complete."""
        cid = bm.headers.get(HDR_CORRELATION_ID)
        if not cid:
            _log.warning("a2a.reply_router.no_correlation_id", extra={"attrs": {"topic": bm.topic}})
            return

        rec = await asyncio.to_thread(self._store.get, cid)
        for _ in range(_LOOKUP_RETRIES):
            if rec is not None:
                break
            await asyncio.sleep(_LOOKUP_DELAY_S)
            rec = await asyncio.to_thread(self._store.get, cid)
        if rec is None:
            _log.warning("a2a.reply_router.unknown_cid", extra={"attrs": {"cid": cid}})
            return

        is_cancel = bm.headers.get(HDR_KIND) == KIND_STEP_CANCELLED
        # step.cancelled also converts records the Supervisor already flagged as
        # dead-lettered (the diagram's DLQ → cancel-A hop); plain replies only
        # ever land on waiting records.
        allowed_from = (STATUS_WAITING, STATUS_DEAD_LETTERED) if is_cancel else (STATUS_WAITING,)
        if rec.get("status") not in allowed_from:
            return  # already resolved/retired — "late replies recognized by ID and ignored"

        if not await self._dedup.claim_reply(cid):
            return  # another consumer/sweep won the race

        if is_cancel:
            result = {"error": f"step cancelled: {bm.headers.get(HDR_ERROR) or 'cancelled by supervisor'}"}
            status = STATUS_CANCELLED
        else:
            try:
                result = {"task": json.loads(bm.value)}
            except Exception as e:
                result = {"error": f"unparseable bus reply: {e}"}
            status = STATUS_RESOLVED

        await asyncio.to_thread(self._store.resolve, cid, status, result=result, allowed_from=allowed_from)
        _log.info("a2a.reply_router.recorded", extra={"attrs": {"cid": cid, "status": status}})
        await self._maybe_resume_group(rec)

    # ------------------------------------------------------------------
    async def _maybe_resume_group(self, rec: dict) -> None:
        """Resume the run exactly once, when its dispatch group has fully resolved."""
        group_id = rec.get("group_id") or rec["_id"]
        pending = await asyncio.to_thread(self._store.pending_in_group, group_id)
        if pending > 0:
            return
        # Group-level claim: two records resolving concurrently must not both invoke.
        if not await self._dedup.claim_group_resume(group_id):
            return
        results = await asyncio.to_thread(self._store.group_results, group_id)
        self._tasks.append(asyncio.create_task(self._resume_fn(rec, results)))
        _log.info("a2a.reply_router.resuming", extra={"attrs": {
            "group": group_id, "thread_id": rec.get("thread_id"), "tasks": list(results),
        }})

    # ------------------------------------------------------------------
    async def _sweep_loop(self) -> None:
        """Time out expired waits so no run can suspend forever (Supervisor-less mode)."""
        interval = get_settings().bus_sweep_interval_seconds
        while True:
            await asyncio.sleep(interval)
            try:
                expired = await asyncio.to_thread(self._store.expired)
                for rec in expired:
                    await self.expire(rec)
            except Exception as e:
                _log.error("a2a.reply_router.sweep_failed", extra={"attrs": {"error": str(e)}}, exc_info=True)

    async def expire(self, rec: dict) -> None:
        """Resolve one expired wait as a timeout; resume its group if now complete."""
        cid = rec["_id"]
        if not await self._dedup.claim_reply(cid):
            return  # a real reply won at the wire
        result = {"error": f"deadline exceeded waiting for agent '{rec.get('agent_id')}'"}
        await asyncio.to_thread(self._store.resolve, cid, STATUS_TIMEOUT, result=result)
        _log.warning("a2a.reply_router.deadline_expired", extra={"attrs": {"cid": cid, "agent": rec.get("agent_id")}})
        await self._maybe_resume_group(rec)

    async def cancel_run(self, thread_id: str, run_id: str) -> int:
        """Cancel every pending wait of one run (the user-facing cancel API).

        Resolves each pending record as cancelled with an error result, then
        group-resumes — the run unblocks immediately, the blackboard records the
        cancellations, and the Gate/Synthesizer produce a partial answer.
        Returns how many waits were cancelled.
        """
        records = await asyncio.to_thread(self._store.waiting_for_run, thread_id, run_id)
        cancelled = 0
        for rec in records:
            cid = rec["_id"]
            if not await self._dedup.claim_reply(cid):
                continue
            result = {"error": "cancelled by user request"}
            ok = await asyncio.to_thread(
                self._store.resolve, cid, STATUS_CANCELLED,
                result=result, allowed_from=(STATUS_WAITING, STATUS_DEAD_LETTERED),
            )
            if ok:
                cancelled += 1
                await self._maybe_resume_group(rec)
        _log.info("a2a.reply_router.run_cancelled", extra={"attrs": {"thread_id": thread_id, "run_id": run_id, "count": cancelled}})
        return cancelled

    # ------------------------------------------------------------------
    async def _resume(self, rec: dict, results: dict) -> None:
        """Resume the suspended run on its durable checkpoint (blocking work on a thread).

        ``results`` is the combined group payload ``{task_id: result}`` the
        Executor's ``interrupt()`` returns. If the run finishes here (no further
        suspension), persists the session messages — the job ``/chat`` does when
        a run completes inline.
        """
        graph = get_graph()
        config = get_thread_config(rec["thread_id"])
        for _ in range(_RESUME_POLLS):
            snap = await asyncio.to_thread(graph.get_state, config)
            if snap and snap.next:
                break
            await asyncio.sleep(_RESUME_POLL_DELAY_S)
        else:
            _log.error("a2a.reply_router.no_pending_interrupt", extra={"attrs": {"thread_id": rec["thread_id"]}})
            return

        result = await asyncio.to_thread(graph.invoke, Command(resume=results), config)
        if result.get("__interrupt__"):
            return  # suspended again on a later wave's bus group — a later resume continues it

        try:
            store = get_mongo_store()
            await store.save_messages(
                rec["thread_id"],
                result.get("messages", []),
                result.get("short_term_memory", []),
            )
        except Exception as e:
            _log.warning("a2a.reply_router.save_messages_failed", extra={"attrs": {"error": str(e)}})
        _log.info(
            "a2a.reply_router.run_completed",
            extra={"attrs": {"thread_id": rec["thread_id"], "run_id": rec.get("run_id")}},
        )
