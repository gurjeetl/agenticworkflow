"""A2A Supervisor: the control plane for asynchronous agent work (Phase 2).

Owns the diagram's failure ladder for every outstanding bus request::

    deadline expired
        ├─ agent heartbeat healthy & extensions left → EXTEND (slow, not dead)
        ├─ attempts left                             → RETRY  (attempt+1, deduped)
        └─ exhausted                                 → DEAD-LETTER (genie.dlq)

and permanently consumes ``genie.dlq`` (poison pills from agents + its own
retry-exhausted letters), producing ``step.cancelled`` control messages to the
reply topic so the waiting run **unblocks immediately** instead of burning its
deadline — the gateway's reply-router converts that into a blackboard error and
the existing Gate → Planner loop re-plans (fallback agent / partial answer).

The Supervisor deliberately cannot resume graphs itself — it has no graph. All
unblocking goes through the bus (control plane), never through tracing.

Deployment: run ``services/supervisor/server.py`` and set
``bus_supervisor_enabled=true`` on the gateway so its simple timeout sweep
stands down (this ladder replaces it). Multiple Supervisor instances are safe:
every transition is an atomic status CAS and every produce is deduped
downstream.

(Naming: this is the diagram's "Orchestrator" box; it ships as *Supervisor* to
avoid colliding with the graph's ``orchestrator.py`` decomposition node.)
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from genie.messaging.awaiting import STATUS_DEAD_LETTERED, STATUS_WAITING, get_awaiting_store
from genie.messaging.broker import BusMessage, get_broker
from genie.messaging.envelope import (
    HDR_ATTEMPT,
    HDR_CORRELATION_ID,
    HDR_DEADLINE,
    HDR_ERROR,
    HDR_FROM,
    HDR_GROUP_ID,
    HDR_KIND,
    HDR_REPLY_TO,
    HDR_RUN_ID,
    HDR_TASK_ID,
    HDR_TENANT_ID,
    HDR_THREAD_ID,
    HDR_TO,
    HDR_TRACE_ID,
    KIND_DEAD_LETTER,
    KIND_REQUEST,
    KIND_STEP_CANCELLED,
    correlation_id,
    dlq_topic,
    reply_topic,
)
from genie.observability import get_logger
from genie.platform.config import get_settings
from genie.registry.registry_client import get_registry_client

_log = get_logger(__name__)


class Supervisor:
    """Deadline ladder + DLQ consumer. Collaborators injectable for tests."""

    def __init__(self, broker=None, awaiting=None, registry=None) -> None:
        """Default to the process singletons; tests pass fakes."""
        self._broker = broker or get_broker()
        self._store = awaiting or get_awaiting_store()
        self._registry = registry or get_registry_client()
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Launch the expiry sweep and the permanent DLQ consumer."""
        self._tasks.append(asyncio.create_task(self._sweep_loop()))
        self._tasks.append(asyncio.create_task(self._dlq_loop()))
        _log.info("a2a.supervisor.started", extra={"attrs": {"dlq": dlq_topic()}})

    async def stop(self) -> None:
        """Cancel both loops."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    # ------------------------------------------------------------------
    # Expiry ladder
    # ------------------------------------------------------------------
    async def _sweep_loop(self) -> None:
        """Walk expired waits on the configured cadence."""
        interval = get_settings().bus_sweep_interval_seconds
        while True:
            await asyncio.sleep(interval)
            try:
                expired = await asyncio.to_thread(self._store.expired)
                for rec in expired:
                    await self.handle_expired(rec)
            except Exception as e:
                _log.error("a2a.supervisor.sweep_failed", extra={"attrs": {"error": str(e)}}, exc_info=True)

    def _extend_ms(self) -> int:
        settings = get_settings()
        return settings.bus_extend_ms or settings.a2a_default_deadline_ms

    async def _agent_healthy(self, agent_id: str) -> bool:
        """Heartbeat check via the registry — present == fresh (the store TTL-filters)."""
        try:
            meta = await asyncio.to_thread(self._registry.get, agent_id)
            return meta is not None
        except Exception:
            return False

    async def handle_expired(self, rec: dict) -> None:
        """Apply the ladder to one expired wait: extend → retry → dead-letter."""
        settings = get_settings()
        cid = rec["_id"]

        if rec.get("extends_used", 0) < settings.bus_max_extends and await self._agent_healthy(rec.get("agent_id", "")):
            new_deadline = datetime.now(timezone.utc) + timedelta(milliseconds=self._extend_ms())
            if await asyncio.to_thread(self._store.extend, cid, new_deadline):
                _log.info("a2a.supervisor.extend_granted", extra={"attrs": {
                    "cid": cid, "agent": rec.get("agent_id"),
                    "extends_used": rec.get("extends_used", 0) + 1, "of": settings.bus_max_extends,
                    "reason": "agent heartbeat healthy — slow, not dead",
                }})
            return

        if rec.get("attempt", 1) < settings.bus_max_attempts and rec.get("request"):
            await self.retry(rec)
            return

        await self.dead_letter(rec, error="no reply before deadline — extensions and retries exhausted")

    async def retry(self, rec: dict) -> None:
        """Re-produce the stored request as attempt+1 (new deterministic cid, same group).

        The old record closes as ``retried`` (no result — its replacement keeps
        the group open); a late reply to the old attempt is ignored by status.
        Safe under concurrent Supervisors: the CAS on the old record elects one
        winner, and the agent's ``(cid, attempt)`` dedup absorbs double-produces.
        """
        old_cid = rec["_id"]
        attempt = rec.get("attempt", 1) + 1
        new_cid = correlation_id(rec.get("run_id", ""), rec.get("task_id", ""), attempt)
        if not await asyncio.to_thread(self._store.mark_retried, old_cid):
            return  # another instance already handled it

        payload = rec["request"]
        try:  # keep the in-body metadata consistent with the new attempt's headers
            data = json.loads(payload)
            data.setdefault("metadata", {})["correlation_id"] = new_cid
            payload = json.dumps(data)
        except Exception:
            pass  # headers are authoritative; a non-JSON body just ships as-is

        deadline = datetime.now(timezone.utc) + timedelta(milliseconds=self._extend_ms())
        await asyncio.to_thread(
            self._store.put, new_cid,
            thread_id=rec.get("thread_id", ""), run_id=rec.get("run_id", ""),
            task_id=rec.get("task_id", ""), agent_id=rec.get("agent_id", ""),
            deadline=deadline, group_id=rec.get("group_id") or old_cid, attempt=attempt,
            request=payload, inbox_topic=rec.get("inbox_topic"), tenant_id=rec.get("tenant_id"),
        )
        await self._broker.produce(
            rec.get("inbox_topic") or "",
            value=payload.encode("utf-8"),
            key=rec.get("thread_id") or None,
            headers={
                HDR_KIND: KIND_REQUEST,
                HDR_CORRELATION_ID: new_cid,
                HDR_ATTEMPT: str(attempt),
                HDR_GROUP_ID: rec.get("group_id") or old_cid,
                HDR_FROM: "supervisor",
                HDR_TO: rec.get("agent_id", ""),
                HDR_REPLY_TO: reply_topic(),
                HDR_THREAD_ID: rec.get("thread_id", ""),
                HDR_RUN_ID: rec.get("run_id", ""),
                HDR_TASK_ID: rec.get("task_id", ""),
                HDR_TRACE_ID: rec.get("run_id", ""),
                HDR_TENANT_ID: rec.get("tenant_id") or "",
                HDR_DEADLINE: deadline.isoformat(),
            },
        )
        _log.warning("a2a.supervisor.retried", extra={"attrs": {
            "old_cid": old_cid, "new_cid": new_cid, "attempt": attempt, "agent": rec.get("agent_id"),
        }})

    async def dead_letter(self, rec: dict, *, error: str) -> None:
        """Park an exhausted wait on the DLQ (full payload preserved, replayable).

        The record flips to ``dead_lettered`` (still blocking its group); the DLQ
        consumer — this very service — then emits ``step.cancelled`` so the
        gateway converts it to a cancelled result and the run unblocks.
        """
        cid = rec["_id"]
        if not await asyncio.to_thread(self._store.mark_dead_lettered, cid):
            return  # another instance won
        await self._broker.produce(
            dlq_topic(),
            value=(rec.get("request") or "").encode("utf-8"),
            key=rec.get("thread_id") or None,
            headers={
                HDR_KIND: KIND_DEAD_LETTER,
                HDR_CORRELATION_ID: cid,
                HDR_ATTEMPT: str(rec.get("attempt", 1)),
                HDR_FROM: "supervisor",
                HDR_TO: rec.get("agent_id", ""),
                HDR_THREAD_ID: rec.get("thread_id", ""),
                HDR_RUN_ID: rec.get("run_id", ""),
                HDR_TASK_ID: rec.get("task_id", ""),
                HDR_ERROR: error,
            },
        )
        _log.error("a2a.supervisor.dead_lettered", extra={"attrs": {"cid": cid, "agent": rec.get("agent_id"), "error": error}})

    # ------------------------------------------------------------------
    # DLQ consumer → step.cancelled (unblock the waiting run immediately)
    # ------------------------------------------------------------------
    async def _dlq_loop(self) -> None:
        """Permanently consume the DLQ (poison pills + retry-exhausted letters)."""
        group = f"{get_settings().bus_topic_prefix}-supervisor"
        async for bm in self._broker.consume([dlq_topic()], group=group):
            try:
                await self.handle_dead_letter(bm)
            except Exception as e:
                _log.error("a2a.supervisor.dlq_handle_failed", extra={"attrs": {"error": str(e)}}, exc_info=True)

    async def handle_dead_letter(self, bm: BusMessage) -> None:
        """Unblock the run waiting on a dead-lettered request via ``step.cancelled``.

        Covers both DLQ inflows: an agent's poison pill (record still
        ``waiting``) and this service's own retry-exhausted letters
        (``dead_lettered``). Records already resolved — or letters replayed by
        an operator — are skipped.
        """
        cid = bm.headers.get(HDR_CORRELATION_ID)
        if not cid:
            return
        rec = await asyncio.to_thread(self._store.get, cid)
        if rec is None or rec.get("status") not in (STATUS_WAITING, STATUS_DEAD_LETTERED):
            return  # unknown/foreign letter, or the wait was already resolved
        error = bm.headers.get(HDR_ERROR) or "dead-lettered"
        await self._broker.produce(
            reply_topic(),
            value=b"{}",
            key=rec.get("thread_id") or None,
            headers={
                HDR_KIND: KIND_STEP_CANCELLED,
                HDR_CORRELATION_ID: cid,
                HDR_FROM: "supervisor",
                HDR_THREAD_ID: rec.get("thread_id", ""),
                HDR_RUN_ID: rec.get("run_id", ""),
                HDR_ERROR: error,
            },
        )
        _log.info("a2a.supervisor.step_cancelled", extra={"attrs": {
            "cid": cid, "error": error, "decision": "unblock caller — gate re-plans",
        }})
