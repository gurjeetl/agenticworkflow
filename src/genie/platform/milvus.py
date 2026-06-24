"""Centralized Milvus connection for the platform.

One process-wide :class:`pymilvus.MilvusClient`, lazily created from
``milvus_uri``/``milvus_db_path``/``milvus_token``. Returns ``None`` when neither a
URI nor a Lite-file path is configured, or when pymilvus is missing / the connection
fails — so the framework runs without Milvus (callers fail open).

Sync only: pymilvus' ``MilvusClient`` has no async client.
"""
from __future__ import annotations

import logging
import os

from genie.platform.config import get_settings

_log = logging.getLogger(__name__)

_client = None
_initialized = False


def _resolve_uri() -> str | None:
    """The Milvus URI: a remote http(s) endpoint OR a local Milvus Lite file path."""
    settings = get_settings()
    return settings.milvus_uri or settings.milvus_db_path


def get_milvus_client():
    """Return the process-wide MilvusClient, or ``None`` when disabled/unavailable."""
    global _client, _initialized
    if _initialized:
        return _client
    _initialized = True

    uri = _resolve_uri()
    if not uri:
        _log.warning("milvus.disabled", extra={"attrs": {"reason": "MILVUS_URI/MILVUS_DB_PATH unset"}})
        return None

    # pymilvus' ORM reads os.environ['MILVUS_URI'] at import and only accepts http(s)
    # URIs — a local file path there would crash `import pymilvus` process-wide.
    # Move any non-http value out of the env before importing.
    env_uri = os.environ.get("MILVUS_URI")
    if env_uri and not env_uri.lower().startswith("http"):
        os.environ.pop("MILVUS_URI", None)

    try:
        from pymilvus import MilvusClient
    except ImportError:
        _log.warning("milvus.disabled", extra={"attrs": {"reason": "pymilvus not installed"}})
        return None
    try:
        _client = MilvusClient(uri=uri, token=get_settings().milvus_token or "")
    except Exception as e:
        _log.warning("milvus.connect_failed", extra={"attrs": {"uri": uri, "error": str(e)}})
        _client = None
    return _client


def close_milvus_client() -> None:
    """Close the Milvus client if connected (errors swallowed). Idempotent."""
    global _client, _initialized
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
    _client = None
    _initialized = False
