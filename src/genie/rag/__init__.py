"""RAG service protocols — the platform-side view of the retrieval boundary.

Defines the structural ``RetrievalService`` / ``IngestionService`` protocols that
both the in-process :class:`~genie.rag.adapters.local.LocalRAGAdapter` and the
HTTP :class:`~genie.rag.adapters.remote.RemoteRAGAdapter` satisfy. Wire models
live in the standalone ``genie_rag_contracts`` package so the RAG service and the
platform agree on the contract without importing each other.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from genie_rag_contracts.retrieval import RetrievalRequest, RetrievalResponse


@runtime_checkable
class RetrievalService(Protocol):
    """Anything that answers a :class:`RetrievalRequest` with a :class:`RetrievalResponse`."""

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Answer a ``RetrievalRequest`` with a ``RetrievalResponse``."""
        ...


@runtime_checkable
class IngestionService(Protocol):
    """Anything that can ingest raw ``content`` into the retrieval corpus."""

    async def ingest(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Ingest raw ``content`` (with optional metadata) into the retrieval corpus."""
        ...


__all__ = ["RetrievalService", "IngestionService"]
