"""Aggregate shutdown for all centralized datastore connections.

One call to :func:`close_all_connections` tears down whichever shared clients/pools
each process actually created (Mongo, Redis, Milvus, Postgres, SQL Server). It is
async because Redis/Postgres/SQL Server have async closes; every backend's close is
idempotent and best-effort, so calling this when nothing was opened is a safe no-op.
"""
from __future__ import annotations

from genie.platform.mongo import close_mongo_clients
from genie.platform.redis import close_redis_clients
from genie.platform.milvus import close_milvus_client
from genie.platform.postgres import close_pg_pools
from genie.platform.sqlserver import close_sqlserver


async def close_all_connections() -> None:
    """Close every shared datastore connection this process opened. Idempotent."""
    await close_redis_clients()
    await close_pg_pools()
    await close_sqlserver()
    close_milvus_client()
    close_mongo_clients()
