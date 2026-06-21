"""Factory that selects the RAG adapter from configuration.

``rag_backend: local`` (the default) returns an in-process
:class:`LocalRAGAdapter`, so nothing breaks when the standalone RAG service is
not running. ``rag_backend: remote`` returns a :class:`RemoteRAGAdapter` pointed
at ``rag_service_url``. Mirrors the registry client's "optional / resilient"
posture — the platform degrades gracefully rather than hard-failing on RAG.
"""
from __future__ import annotations

from genie.observability import get_logger
from genie.platform.config import get_settings
from genie.rag.adapters.local import LocalRAGAdapter
from genie.rag.adapters.remote import RemoteRAGAdapter

_log = get_logger(__name__)


def get_rag_adapter() -> LocalRAGAdapter | RemoteRAGAdapter:
    """Return the configured RAG adapter (local by default, remote when selected)."""
    s = get_settings()
    backend = (s.rag_backend or "local").strip().lower()
    if backend == "remote":
        _log.info("rag.adapter_remote", extra={"attrs": {"url": s.rag_service_url}})
        return RemoteRAGAdapter(
            base_url=s.rag_service_url,
            timeout=s.rag_service_timeout_s,
            api_key=s.rag_service_auth_token or None,
        )
    _log.info("rag.adapter_local", extra={"attrs": {"backend": backend}})
    return LocalRAGAdapter()
