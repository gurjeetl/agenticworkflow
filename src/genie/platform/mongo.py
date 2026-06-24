"""Centralized MongoDB connections for the platform.

One process-wide async (motor) client and one sync (pymongo) client, lazily created
from the central Settings (``mongodb_uri``/``mongodb_db``) and shared by every store,
agent, and tool that needs MongoDB — so a connection is configured once and reused
everywhere instead of each component constructing its own client.

Use the SYNC handle (:func:`get_sync_mongo_db`) from synchronous code — LangGraph
nodes, agents, and MCP tools. A motor (async) client is bound to the event loop it is
created on and fails ("attached to a different loop") when used from another, so async
access (:func:`get_async_mongo_db`) is only safe from the gateway's event loop.
"""
from __future__ import annotations

import motor.motor_asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from pymongo.database import Database

from genie.platform.config import get_settings

_async_client: AsyncIOMotorClient | None = None
_sync_client: MongoClient | None = None


def get_async_mongo_client() -> AsyncIOMotorClient:
    """Return the process-wide async (motor) client, creating it on first use."""
    global _async_client
    if _async_client is None:
        _async_client = motor.motor_asyncio.AsyncIOMotorClient(get_settings().mongodb_uri)
    return _async_client


def get_async_mongo_db(db_name: str | None = None) -> AsyncIOMotorDatabase:
    """Return an async database handle (defaults to the configured ``mongodb_db``)."""
    return get_async_mongo_client()[db_name or get_settings().mongodb_db]


def get_sync_mongo_client() -> MongoClient:
    """Return the process-wide sync (pymongo) client, creating it on first use."""
    global _sync_client
    if _sync_client is None:
        _sync_client = MongoClient(get_settings().mongodb_uri)
    return _sync_client


def get_sync_mongo_db(db_name: str | None = None) -> Database:
    """Return a sync database handle (defaults to the configured ``mongodb_db``)."""
    return get_sync_mongo_client()[db_name or get_settings().mongodb_db]


def close_mongo_clients() -> None:
    """Close whichever shared clients were created. Idempotent; safe at shutdown."""
    global _async_client, _sync_client
    if _async_client is not None:
        _async_client.close()
        _async_client = None
    if _sync_client is not None:
        try:
            _sync_client.close()
        except Exception:
            pass
        _sync_client = None
