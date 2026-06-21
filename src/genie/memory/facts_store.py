"""Sync (pymongo) MongoDB store for per-thread structured facts (the
``agent_facts`` collection): global facts recalled every session and session
facts with a sliding TTL. Read/written by the Planner and Synthesizer nodes."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo import ASCENDING
from pymongo.errors import PyMongoError

from genie.memory.commit_store import get_commit_store
from genie.platform.config import get_settings

_log = logging.getLogger(__name__)

# Session facts get a sliding TTL: every read (Planner) and write (Synthesizer)
# pushes expiry 30 days out, so an actively-resumed conversation never loses its
# facts — even a multi-day gap is well inside the window — while truly abandoned
# threads (e.g. the fresh-UUID threads the trace UI mints per page load) self-clean.
_SESSION_TTL = timedelta(days=30)


class FactsStore:
    """MongoDB-backed structured fact memory, distinct from the ``agent_commits``
    audit log.

    Two scopes share one ``agent_facts`` collection:
      - ``global``  — stable user/world facts, keyed by entity, recalled in every
        session. Never expire.
      - ``session`` — facts only meaningful inside one conversation, keyed by
        ``thread_id``. Carry a sliding TTL (see ``_SESSION_TTL``).

    Sync on purpose (pymongo, not motor): the Planner and Synthesizer that call
    this are sync LangGraph nodes. Reuses the commit store's MongoClient so the
    whole sync side shares one connection pool. Writes are best-effort — a failure
    is logged and swallowed so a turn never crashes on a persistence error.
    """

    def __init__(self) -> None:
        # Reuse the commit store's pymongo client (one pool for the sync side).
        client = get_commit_store()._client
        db_name = get_settings().mongodb_db
        self._facts = client[db_name]["agent_facts"]

    @property
    def enabled(self) -> bool:
        """Always True — MongoDB is the required primary datastore."""
        return True

    def ensure_indexes(self) -> None:
        """Create the scope/thread lookup indexes and the session-only TTL index.
        Best-effort: failures are logged and swallowed."""
        try:
            self._facts.create_index([("thread_id", ASCENDING)])
            self._facts.create_index([("scope", ASCENDING)])
            # Session-only TTL: expire at the `expireAt` instant. The partial filter
            # scopes the TTL index to session docs, so global docs are immune even
            # though Mongo TTL is per-collection (belt-and-suspenders: globals never
            # set `expireAt`, and a TTL index ignores docs lacking the field).
            self._facts.create_index(
                [("expireAt", ASCENDING)],
                expireAfterSeconds=0,
                partialFilterExpression={"scope": "session"},
            )
        except PyMongoError as e:
            _log.warning("facts_store.index_failed", extra={"attrs": {"error": str(e)}})

    def upsert(
        self,
        scope: str,
        key: str,
        value: str,
        *,
        entity: str | None = None,
        thread_id: str | None = None,
        run_id: str = "",
    ) -> None:
        """Insert or update one fact. Global facts are keyed by ``key`` and clear any
        stale TTL; session facts are keyed by ``thread_id``+``key`` and (re)set the
        sliding ``expireAt``. Best-effort: failures are logged and swallowed."""
        now = datetime.now(timezone.utc)
        try:
            if scope == "global":
                self._facts.update_one(
                    {"_id": f"g::{key}"},
                    {
                        "$set": {
                            "scope": "global",
                            "key": key,
                            "value": value,
                            "entity": entity or key,
                            "thread_id": None,
                            "run_id": run_id,
                            "updated_at": now,
                        },
                        # Clear any stale TTL if a key was previously session-scoped.
                        "$unset": {"expireAt": ""},
                    },
                    upsert=True,
                )
            else:
                self._facts.update_one(
                    {"_id": f"s::{thread_id}::{key}"},
                    {
                        "$set": {
                            "scope": "session",
                            "key": key,
                            "value": value,
                            "entity": None,
                            "thread_id": thread_id,
                            "run_id": run_id,
                            "updated_at": now,
                            "expireAt": now + _SESSION_TTL,
                        }
                    },
                    upsert=True,
                )
        except PyMongoError as e:
            _log.warning(
                "facts_store.upsert_failed",
                extra={"attrs": {"scope": scope, "key": key, "error": str(e)}},
            )

    def query(self, thread_id: str) -> dict[str, str]:
        """Merged facts visible to this thread: all globals plus this thread's
        session facts (session overrides global on a key collision). Slides the
        TTL on the thread's session facts so an active conversation stays alive.
        """
        out: dict[str, str] = {}
        try:
            for d in self._facts.find({"scope": "global"}):
                out[d["key"]] = d["value"]
            for d in self._facts.find({"scope": "session", "thread_id": thread_id}):
                out[d["key"]] = d["value"]
            if thread_id:
                self._facts.update_many(
                    {"scope": "session", "thread_id": thread_id},
                    {"$set": {"expireAt": datetime.now(timezone.utc) + _SESSION_TTL}},
                )
        except PyMongoError as e:
            _log.warning("facts_store.query_failed", extra={"attrs": {"error": str(e)}})
        return out


_store: FactsStore | None = None


def get_facts_store() -> FactsStore:
    """Return the process-wide FactsStore singleton, creating it on first use."""
    global _store
    if _store is None:
        _store = FactsStore()
    return _store
