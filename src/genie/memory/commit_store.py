from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, MongoClient
from pymongo.errors import PyMongoError

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
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "agent_memory")
        self._client = MongoClient(uri)
        db = self._client[db_name]
        self._commits = db["agent_commits"]
        self._links = db["entity_links"]

    @property
    def enabled(self) -> bool:
        return True

    def ensure_indexes(self) -> None:
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

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


_store: MongoCommitStore | None = None


def get_commit_store() -> MongoCommitStore:
    global _store
    if _store is None:
        _store = MongoCommitStore()
    return _store
