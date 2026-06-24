"""Centralized SQL Server connections for the platform (pyodbc + aioodbc).

Sync access via pyodbc (the ODBC driver manager pools connections), async access via
an aioodbc pool — mirroring the sync/async split used elsewhere: use the sync helper
from LangGraph nodes / agents / tools, the async helpers from the gateway's event loop.

Built from ``sqlserver_dsn`` — a full ODBC connection string, e.g.::

    DRIVER={ODBC Driver 18 for SQL Server};SERVER=host,1433;DATABASE=db;UID=user;PWD=pass;TrustServerCertificate=yes

Requires the Microsoft "ODBC Driver 18 for SQL Server" installed on the host OS.
Optional backend: a missing ``sqlserver_dsn`` raises a clear "not configured" error.
Driver imports are lazy so the module loads even before pyodbc/aioodbc are installed.

Usage example — the canonical pattern a tool/agent copies (see ``sqlserver_healthcheck``):

    with get_sqlserver_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager

from genie.platform.config import get_settings

_log = logging.getLogger(__name__)

_POOL_MIN = 1
_POOL_MAX = 10

_async_pool = None


def _dsn() -> str:
    """Return the configured SQL Server ODBC connection string, or raise when unset."""
    dsn = get_settings().sqlserver_dsn
    if not dsn:
        raise RuntimeError("SQL Server is not configured (set sqlserver_dsn).")
    return dsn


@contextmanager
def get_sqlserver_connection():
    """Acquire a sync pyodbc connection (context manager; ODBC manager pools these)."""
    dsn = _dsn()  # validate config before importing the driver
    import pyodbc

    conn = pyodbc.connect(dsn)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


async def get_async_sqlserver_pool():
    """Return the process-wide aioodbc pool, creating it on first use."""
    global _async_pool
    if _async_pool is None:
        dsn = _dsn()  # validate config before importing the driver
        import aioodbc

        _async_pool = await aioodbc.create_pool(
            dsn=dsn, minsize=_POOL_MIN, maxsize=_POOL_MAX, autocommit=True
        )
    return _async_pool


@asynccontextmanager
async def get_async_sqlserver_connection():
    """Acquire an async connection from the shared aioodbc pool (async context manager)."""
    pool = await get_async_sqlserver_pool()
    async with pool.acquire() as conn:
        yield conn


def sqlserver_healthcheck() -> bool:
    """Run ``SELECT 1`` over a sync connection; True on success. Example usage pattern."""
    with get_sqlserver_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1")
        return cur.fetchone()[0] == 1


async def sqlserver_healthcheck_async() -> bool:
    """Run ``SELECT 1`` over an async connection; True on success. Example usage pattern."""
    async with get_async_sqlserver_connection() as conn:
        cur = await conn.cursor()
        await cur.execute("SELECT 1")
        row = await cur.fetchone()
        return row[0] == 1


async def close_sqlserver() -> None:
    """Close the async pool (errors swallowed). Idempotent; safe at shutdown.

    Sync pyodbc connections are closed per-use by ``get_sqlserver_connection``;
    there is no process-wide sync pool to tear down here.
    """
    global _async_pool
    if _async_pool is not None:
        try:
            _async_pool.close()
            await _async_pool.wait_closed()
        except Exception:
            pass
        _async_pool = None
