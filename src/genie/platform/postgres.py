"""Centralized PostgreSQL connections for the platform (psycopg v3 + psycopg_pool).

One process-wide sync pool and one async pool, lazily created from ``postgres_dsn``
(a libpq URI, e.g. ``postgresql://user:pass@host:5432/db``). psycopg v3 gives sync,
async, and pooling in a single driver, mirroring the sync/async split used elsewhere:
use the sync helpers from LangGraph nodes / agents / tools, the async helpers from the
gateway's event loop.

Optional backend: a missing ``postgres_dsn`` raises a clear "not configured" error
(callers that explicitly ask for Postgres want it). Driver imports are lazy so the
module loads even before ``psycopg``/``psycopg_pool`` are installed.

Usage example — the canonical pattern a tool/agent copies (see ``postgres_healthcheck``):

    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager

from genie.platform.config import get_settings

_log = logging.getLogger(__name__)

# Conservative pool sizing; expose via config later if a workload needs tuning.
_POOL_MIN = 1
_POOL_MAX = 10

_sync_pool = None
_async_pool = None


def _dsn() -> str:
    """Return the configured Postgres DSN, or raise a clear error when unset."""
    dsn = get_settings().postgres_dsn
    if not dsn:
        raise RuntimeError("PostgreSQL is not configured (set postgres_dsn).")
    return dsn


def get_pg_pool():
    """Return the process-wide sync connection pool, creating+opening it on first use."""
    global _sync_pool
    if _sync_pool is None:
        dsn = _dsn()  # validate config before importing the driver
        from psycopg_pool import ConnectionPool

        _sync_pool = ConnectionPool(
            dsn, min_size=_POOL_MIN, max_size=_POOL_MAX, open=True
        )
    return _sync_pool


@contextmanager
def get_pg_connection():
    """Acquire a sync connection from the shared pool (context manager)."""
    with get_pg_pool().connection() as conn:
        yield conn


async def get_async_pg_pool():
    """Return the process-wide async connection pool, creating+opening it on first use."""
    global _async_pool
    if _async_pool is None:
        dsn = _dsn()  # validate config before importing the driver
        from psycopg_pool import AsyncConnectionPool

        pool = AsyncConnectionPool(
            dsn, min_size=_POOL_MIN, max_size=_POOL_MAX, open=False
        )
        await pool.open()
        _async_pool = pool
    return _async_pool


@asynccontextmanager
async def get_async_pg_connection():
    """Acquire an async connection from the shared pool (async context manager)."""
    pool = await get_async_pg_pool()
    async with pool.connection() as conn:
        yield conn


def postgres_healthcheck() -> bool:
    """Run ``SELECT 1`` over a sync connection; True on success. Example usage pattern."""
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone()[0] == 1


async def postgres_healthcheck_async() -> bool:
    """Run ``SELECT 1`` over an async connection; True on success. Example usage pattern."""
    async with get_async_pg_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            row = await cur.fetchone()
            return row[0] == 1


async def close_pg_pools() -> None:
    """Close both pools (errors swallowed). Idempotent; safe at shutdown."""
    global _sync_pool, _async_pool
    if _sync_pool is not None:
        try:
            _sync_pool.close()
        except Exception:
            pass
        _sync_pool = None
    if _async_pool is not None:
        try:
            await _async_pool.close()
        except Exception:
            pass
        _async_pool = None
