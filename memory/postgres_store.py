from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

_log = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS agent_commits (
    id           BIGSERIAL PRIMARY KEY,
    run_id       TEXT NOT NULL,
    thread_id    TEXT NOT NULL,
    agent_id     TEXT NOT NULL,
    agent_version TEXT NOT NULL,
    task_id      TEXT NOT NULL,
    payload      JSONB NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS agent_commits_run_idx ON agent_commits(run_id);
CREATE INDEX IF NOT EXISTS agent_commits_thread_idx ON agent_commits(thread_id);

CREATE TABLE IF NOT EXISTS entity_links (
    entity_a       TEXT NOT NULL,
    entity_b       TEXT NOT NULL,
    link_type      TEXT NOT NULL,
    source_run_id  TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (entity_a, entity_b, link_type)
);
"""


class PostgresStore:
    """asyncpg-backed durable commit store.

    No-ops when POSTGRES_DSN is unset or asyncpg is missing — like RedisStore,
    keeps the framework usable for dev without standing up Postgres.
    """

    def __init__(self) -> None:
        self._dsn = os.getenv("POSTGRES_DSN")
        self._pool = None
        if not self._dsn:
            _log.warning("postgres.disabled", extra={"attrs": {"reason": "POSTGRES_DSN unset"}})
            return
        try:
            import asyncpg  # noqa: F401
        except ImportError:
            self._dsn = None
            _log.warning("postgres.disabled", extra={"attrs": {"reason": "asyncpg not installed"}})

    @property
    def enabled(self) -> bool:
        return self._dsn is not None

    async def ensure_pool(self) -> None:
        if not self._dsn or self._pool is not None:
            return
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
            async with self._pool.acquire() as con:
                await con.execute(_SCHEMA_SQL)
        except Exception as e:
            _log.warning("postgres.init_failed", extra={"attrs": {"error": str(e)}})
            self._pool = None

    async def commit(
        self,
        run_id: str,
        thread_id: str,
        agent_id: str,
        agent_version: str,
        task_id: str,
        payload: dict[str, Any],
    ) -> None:
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as con:
                await con.execute(
                    "INSERT INTO agent_commits(run_id, thread_id, agent_id, agent_version, task_id, payload, committed_at)"
                    " VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7)",
                    run_id,
                    thread_id,
                    agent_id,
                    agent_version,
                    task_id,
                    json.dumps(payload, default=str),
                    datetime.now(timezone.utc),
                )
        except Exception as e:
            _log.warning(
                "postgres.commit_failed",
                extra={"attrs": {"run_id": run_id, "agent_id": agent_id, "error": str(e)}},
            )

    async def link_entities(
        self,
        entity_a: str,
        entity_b: str,
        link_type: str,
        source_run_id: str,
    ) -> None:
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as con:
                await con.execute(
                    "INSERT INTO entity_links(entity_a, entity_b, link_type, source_run_id)"
                    " VALUES ($1,$2,$3,$4)"
                    " ON CONFLICT (entity_a, entity_b, link_type) DO NOTHING",
                    entity_a,
                    entity_b,
                    link_type,
                    source_run_id,
                )
        except Exception as e:
            _log.warning("postgres.link_failed", extra={"attrs": {"error": str(e)}})

    async def close(self) -> None:
        if self._pool:
            try:
                await self._pool.close()
            except Exception:
                pass
            self._pool = None


_store: PostgresStore | None = None


def get_postgres_store() -> PostgresStore:
    global _store
    if _store is None:
        _store = PostgresStore()
    return _store
