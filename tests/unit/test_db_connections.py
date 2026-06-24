"""Tests for the centralized platform DB connection modules.

No live servers: Mongo/motor clients construct lazily (no connect), and the optional
backends are exercised on their disabled / not-configured paths, so nothing here needs
a running database.
"""
from types import SimpleNamespace

import pytest

from genie.platform import milvus, postgres, sqlserver
from genie.platform import mongo
from genie.platform import redis as predis
from genie.platform.config import get_settings
from genie.platform.db import close_all_connections


def test_mongo_clients_are_singletons_and_reset_on_close():
    mongo.close_mongo_clients()
    a1 = mongo.get_async_mongo_client()
    s1 = mongo.get_sync_mongo_client()
    assert mongo.get_async_mongo_client() is a1
    assert mongo.get_sync_mongo_client() is s1
    mongo.close_mongo_clients()
    assert mongo.get_async_mongo_client() is not a1
    assert mongo.get_sync_mongo_client() is not s1
    mongo.close_mongo_clients()


def test_mongo_db_name_default_and_override():
    assert mongo.get_sync_mongo_db().name == get_settings().mongodb_db
    assert mongo.get_sync_mongo_db("custom_db").name == "custom_db"
    mongo.close_mongo_clients()


def test_redis_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(predis, "get_settings", lambda: SimpleNamespace(redis_url=None))
    assert predis.redis_enabled() is False
    assert predis.get_sync_redis_client() is None
    assert predis.get_async_redis_client() is None  # no loop + disabled


def test_milvus_disabled_returns_none(monkeypatch):
    milvus.close_milvus_client()
    monkeypatch.setattr(
        milvus, "get_settings",
        lambda: SimpleNamespace(milvus_uri=None, milvus_db_path=None, milvus_token=None),
    )
    assert milvus.get_milvus_client() is None
    # cached disabled result until close resets it
    assert milvus.get_milvus_client() is None
    milvus.close_milvus_client()


def test_postgres_not_configured_raises(monkeypatch):
    monkeypatch.setattr(postgres, "get_settings", lambda: SimpleNamespace(postgres_dsn=None))
    with pytest.raises(RuntimeError, match="not configured"):
        postgres.get_pg_pool()
    with pytest.raises(RuntimeError, match="not configured"):
        postgres.postgres_healthcheck()


async def test_postgres_async_not_configured_raises(monkeypatch):
    monkeypatch.setattr(postgres, "get_settings", lambda: SimpleNamespace(postgres_dsn=None))
    with pytest.raises(RuntimeError, match="not configured"):
        await postgres.get_async_pg_pool()


def test_sqlserver_not_configured_raises(monkeypatch):
    monkeypatch.setattr(sqlserver, "get_settings", lambda: SimpleNamespace(sqlserver_dsn=None))
    with pytest.raises(RuntimeError, match="not configured"):
        with sqlserver.get_sqlserver_connection():
            pass


async def test_close_all_connections_idempotent():
    # Safe to call when little/nothing was opened, and twice in a row.
    await close_all_connections()
    await close_all_connections()
