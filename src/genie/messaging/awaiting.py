"""Durable ``a2a_awaiting`` records: which suspended run awaits which reply.

One document per outstanding delivery attempt, written **before** the request is
produced (so a reply can never race an unrecorded wait). Each record belongs to
a **group** — all bus dispatches of one Executor wave — and the run resumes only
when every record in the group carries a result (reply, timeout, or cancel).

Lifecycle::

    waiting ──reply──────────────▶ resolved   (result = {"task": ...})
    waiting ──deadline sweep─────▶ timeout    (result = {"error": ...})
    waiting ──supervisor retry───▶ retried    (replaced by a new record, attempt+1)
    waiting ──supervisor DLQ─────▶ dead_lettered ──step.cancelled──▶ cancelled
    waiting ──user cancel────────▶ cancelled  (result = {"error": ...})

``retried`` records carry no result and are excluded from group completion —
their replacement record (same ``task_id``, same ``group_id``) takes over.
The stored ``request`` payload + ``inbox_topic`` are what let the Supervisor
re-produce a retry and dead-letter the original message with full history.

Sync pymongo on purpose: the Executor writes from LangGraph's synchronous node
thread, and async callers go through ``asyncio.to_thread`` — motor clients are
loop-bound and the Executor runs on transient loops (same reasoning as
``genie.platform.mongo``).
"""
from __future__ import annotations

from datetime import datetime, timezone

from pymongo import ASCENDING
from pymongo.collection import Collection

from genie.platform.mongo import get_sync_mongo_db

STATUS_WAITING = "waiting"
STATUS_RESOLVED = "resolved"
STATUS_TIMEOUT = "timeout"
STATUS_RETRIED = "retried"
STATUS_DEAD_LETTERED = "dead_lettered"
STATUS_CANCELLED = "cancelled"

# Statuses that keep a group open (no resume yet): still waiting for a reply,
# or dead-lettered and waiting for the step.cancelled hop to convert it.
PENDING_STATUSES = (STATUS_WAITING, STATUS_DEAD_LETTERED)

# Records auto-expire well after any sane deadline so the collection can't grow
# unbounded even if a resolve is missed; 7 days keeps them inspectable.
_EXPIRE_AFTER_SECONDS = 7 * 24 * 3600


class AwaitingStore:
    """CRUD over the ``a2a_awaiting`` collection (documents keyed by correlation id)."""

    def __init__(self, collection: Collection | None = None) -> None:
        """Bind to the given collection (tests) or the platform database."""
        self._col = collection if collection is not None else get_sync_mongo_db()["a2a_awaiting"]
        self._indexed = False

    def ensure_indexes(self) -> None:
        """Sweep index (status+deadline), group index, TTL cleanup. Idempotent."""
        if self._indexed:
            return
        self._col.create_index([("status", ASCENDING), ("deadline", ASCENDING)])
        self._col.create_index([("group_id", ASCENDING), ("status", ASCENDING)])
        self._col.create_index([("run_id", ASCENDING), ("status", ASCENDING)])
        self._col.create_index("created_at", expireAfterSeconds=_EXPIRE_AFTER_SECONDS)
        self._indexed = True

    def put(
        self,
        cid: str,
        *,
        thread_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
        deadline: datetime,
        group_id: str,
        attempt: int = 1,
        request: str | None = None,
        inbox_topic: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """Upsert the awaiting record for ``cid`` (idempotent — dispatch re-runs on resume).

        ``request`` (the serialized Message value) + ``inbox_topic`` enable the
        Supervisor's retry and dead-letter produces without re-reading Kafka.
        """
        self.ensure_indexes()
        self._col.update_one(
            {"_id": cid},
            {"$setOnInsert": {
                "_id": cid,
                "thread_id": thread_id,
                "run_id": run_id,
                "task_id": task_id,
                "agent_id": agent_id,
                "deadline": deadline,
                "group_id": group_id,
                "attempt": attempt,
                "extends_used": 0,
                "request": request,
                "inbox_topic": inbox_topic,
                "tenant_id": tenant_id,
                "status": STATUS_WAITING,
                "result": None,
                "created_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )

    def get(self, cid: str) -> dict | None:
        """The awaiting record for ``cid``, or None."""
        return self._col.find_one({"_id": cid})

    def resolve(
        self,
        cid: str,
        status: str = STATUS_RESOLVED,
        *,
        result: dict | None = None,
        allowed_from: tuple[str, ...] = (STATUS_WAITING,),
    ) -> bool:
        """Atomically transition ``allowed_from`` → ``status`` storing the result.

        True only for the single winner — the concurrency guard between a real
        reply, the timeout sweep, a cancel, and the step.cancelled hop
        (which converts ``dead_lettered`` records, hence ``allowed_from``).
        """
        res = self._col.update_one(
            {"_id": cid, "status": {"$in": list(allowed_from)}},
            {"$set": {"status": status, "result": result, "resolved_at": datetime.now(timezone.utc)}},
        )
        return res.modified_count == 1

    def extend(self, cid: str, new_deadline: datetime) -> bool:
        """Grant one deadline extension (Supervisor: agent heartbeat still healthy)."""
        res = self._col.update_one(
            {"_id": cid, "status": STATUS_WAITING},
            {"$set": {"deadline": new_deadline}, "$inc": {"extends_used": 1}},
        )
        return res.modified_count == 1

    def mark_retried(self, cid: str) -> bool:
        """Close a record whose attempt is being superseded by a retry record."""
        res = self._col.update_one(
            {"_id": cid, "status": STATUS_WAITING},
            {"$set": {"status": STATUS_RETRIED, "resolved_at": datetime.now(timezone.utc)}},
        )
        return res.modified_count == 1

    def mark_dead_lettered(self, cid: str) -> bool:
        """Flag a record as dead-lettered; the step.cancelled hop converts it to cancelled."""
        res = self._col.update_one(
            {"_id": cid, "status": STATUS_WAITING},
            {"$set": {"status": STATUS_DEAD_LETTERED, "dead_lettered_at": datetime.now(timezone.utc)}},
        )
        return res.modified_count == 1

    def expired(self, now: datetime | None = None) -> list[dict]:
        """All still-waiting records whose deadline has passed (for the sweeps)."""
        now = now or datetime.now(timezone.utc)
        return list(self._col.find({"status": STATUS_WAITING, "deadline": {"$lt": now}}))

    # ------------------------------------------------------------------
    # Group (fan-out) queries — one group = one Executor wave's bus dispatches.
    # ------------------------------------------------------------------
    def pending_in_group(self, group_id: str) -> int:
        """How many records in the group still block the resume."""
        return self._col.count_documents({"group_id": group_id, "status": {"$in": list(PENDING_STATUSES)}})

    def group_results(self, group_id: str) -> dict[str, dict]:
        """``{task_id: result}`` for every record in the group that carries a result."""
        out: dict[str, dict] = {}
        for rec in self._col.find({"group_id": group_id, "result": {"$ne": None}}):
            out[rec["task_id"]] = rec["result"]
        return out

    def waiting_for_run(self, thread_id: str, run_id: str) -> list[dict]:
        """Every pending record of one run (the cancel API sweeps these)."""
        return list(self._col.find({
            "thread_id": thread_id,
            "run_id": run_id,
            "status": {"$in": list(PENDING_STATUSES)},
        }))


_store: AwaitingStore | None = None


def get_awaiting_store() -> AwaitingStore:
    """Process-wide AwaitingStore singleton."""
    global _store
    if _store is None:
        _store = AwaitingStore()
    return _store


def set_awaiting_store(store: AwaitingStore | None) -> None:
    """Inject a store (tests) or reset to lazy default with ``None``."""
    global _store
    _store = store
