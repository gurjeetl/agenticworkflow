"""Sync (pymongo) MongoDB store for durable output commits and the audit log:
each agent's persistable output in ``agent_commits`` and entity relationships in
``entity_links``. Written by the Synthesizer node. MongoDB is required."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING
from pymongo.errors import PyMongoError

from genie.platform.mongo import get_sync_mongo_db

_log = logging.getLogger(__name__)


class MongoCommitStore:
    """MongoDB-backed durable commit store (replaces the old PostgresStore).

    Synchronous on purpose (pymongo, not motor): the Synthesizer is a sync
    LangGraph node that calls this directly. An async/motor client would be
    bound to the event loop it was created on and blow up with "attached to a
    different loop" when invoked from the graph's loop.

    Persists each agent's persistable output as a document in ``agent_commits``
    and entity relationships in ``entity_links``. Always enabled — MongoDB is the
    framework's primary datastore. Writes are best-effort: a failure is logged
    and swallowed so a synthesis never crashes on a persistence error.
    """

    def __init__(self) -> None:
        """Bind the ``agent_commits`` / ``entity_links`` collections off the shared sync client."""
        db = get_sync_mongo_db()
        self._commits = db["agent_commits"]
        self._links = db["entity_links"]

    @property
    def enabled(self) -> bool:
        """Always True — MongoDB is the required primary datastore."""
        return True

    def ensure_indexes(self) -> None:
        """Create the run/thread lookup indexes and the unique entity-link index.
        Best-effort: failures are logged and swallowed."""
        try:
            self._commits.create_index([("run_id", ASCENDING)])
            self._commits.create_index([("thread_id", ASCENDING)])
            # Mirror the Postgres UNIQUE (entity_a, entity_b, link_type).
            self._links.create_index(
                [("entity_a", ASCENDING), ("entity_b", ASCENDING), ("link_type", ASCENDING)],
                unique=True,
            )
        except PyMongoError as e:
            _log.warning("commit_store.index_failed", extra={"attrs": {"error": str(e)}})

    def commit(
        self,
        run_id: str,
        thread_id: str,
        agent_id: str,
        agent_version: str,
        task_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Append one agent output to the ``agent_commits`` audit log, timestamped.
        Best-effort: failures are logged and swallowed so synthesis never crashes."""
        try:
            self._commits.insert_one(
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "agent_id": agent_id,
                    "agent_version": agent_version,
                    "task_id": task_id,
                    "payload": payload,
                    "committed_at": datetime.now(timezone.utc),
                }
            )
        except PyMongoError as e:
            _log.warning(
                "commit_store.commit_failed",
                extra={"attrs": {"run_id": run_id, "agent_id": agent_id, "error": str(e)}},
            )

    def link_entities(
        self,
        entity_a: str,
        entity_b: str,
        link_type: str,
        source_run_id: str,
    ) -> None:
        """Record a directed relationship between two entities, idempotently (a
        repeat (a, b, type) is a no-op). Best-effort: failures are logged."""
        try:
            # Upsert with $setOnInsert mirrors Postgres ON CONFLICT DO NOTHING.
            self._links.update_one(
                {"entity_a": entity_a, "entity_b": entity_b, "link_type": link_type},
                {
                    "$setOnInsert": {
                        "entity_a": entity_a,
                        "entity_b": entity_b,
                        "link_type": link_type,
                        "source_run_id": source_run_id,
                        "created_at": datetime.now(timezone.utc),
                    }
                },
                upsert=True,
            )
        except PyMongoError as e:
            _log.warning("commit_store.link_failed", extra={"attrs": {"error": str(e)}})


_store: MongoCommitStore | None = None


def get_commit_store() -> MongoCommitStore:
    """Return the process-wide MongoCommitStore singleton, creating it on first use."""
    global _store
    if _store is None:
        _store = MongoCommitStore()
    return _store
