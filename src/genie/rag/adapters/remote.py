"""RemoteRAGAdapter — HTTP client for the standalone RAG service.

Implements both the ``RetrievalService`` and ``IngestionService`` protocols by
calling the RAG service over HTTP, using the path constants from
``genie_rag_contracts.api`` so client and server agree on URLs. On a network /
server error it degrades to an empty response (``retrieval_available=False``) so
the main pipeline keeps running instead of crashing.
"""
from __future__ import annotations

import uuid
from typing import Any

import httpx

from genie_rag_contracts.api import INGEST_BASE_PATH, RETRIEVE_PATH
from genie_rag_contracts.ingestion import IngestJobStatus, IngestRequest
from genie_rag_contracts.retrieval import RetrievalRequest, RetrievalResponse, RetrievalResult

from genie.observability import get_logger

_log = get_logger(__name__)


class RemoteRAGAdapter:
    """HTTP client for the standalone RAG service ([services/rag/server.py])."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        api_key: str | None = None,
    ) -> None:
        """Configure the RAG service base URL, request timeout, retry count, and optional API key."""
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(base_url=self._base_url, headers=headers, timeout=timeout)

    # ── RetrievalService ──────────────────────────────────────────────────────

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """POST the query to the RAG service; empty + unavailable on failure."""
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.post(RETRIEVE_PATH, json=request.model_dump())
                resp.raise_for_status()
                data = resp.json()
                results = [RetrievalResult(**r) for r in data.get("results", [])]
                return RetrievalResponse(
                    results=results,
                    query=request.query,
                    correlation_id=request.correlation_id,
                    retrieval_available=True,
                )
            except httpx.HTTPStatusError as exc:
                _log.warning(
                    "remote_rag.http_error",
                    extra={"attrs": {"status": exc.response.status_code, "attempt": attempt + 1}},
                )
                if exc.response.status_code < 500:
                    break  # client error won't fix itself on retry
            except Exception as exc:
                _log.warning(
                    "remote_rag.error",
                    extra={"attrs": {"error": str(exc), "attempt": attempt + 1}},
                )
        return RetrievalResponse(
            results=[],
            query=request.query,
            correlation_id=request.correlation_id,
            retrieval_available=False,
        )

    # ── IngestionService ──────────────────────────────────────────────────────

    async def ingest(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Ingest raw content via the RAG service's ``/ingest/content`` endpoint."""
        meta = metadata or {}
        doc_id = meta.get("document_id", str(uuid.uuid4()))
        payload = {"document_id": doc_id, "content": content, "metadata": meta}
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.post(f"{INGEST_BASE_PATH}/content", json=payload)
                resp.raise_for_status()
                _log.debug("remote_rag.ingested", extra={"attrs": {"doc_id": doc_id}})
                return
            except Exception as exc:
                _log.warning(
                    "remote_rag.ingest_error",
                    extra={"attrs": {"error": str(exc), "attempt": attempt + 1}},
                )

    async def ingest_request(self, request: IngestRequest) -> IngestJobStatus:
        """Ingest a document by path via the RAG service's ``/ingest/file`` endpoint."""
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.post(f"{INGEST_BASE_PATH}/file", json=request.model_dump())
                resp.raise_for_status()
                return IngestJobStatus(**resp.json())
            except Exception as exc:
                _log.warning(
                    "remote_rag.ingest_request_error",
                    extra={"attrs": {"error": str(exc), "attempt": attempt + 1}},
                )
        return IngestJobStatus(
            job_id="",
            document_id=request.document_id,
            status="failed",
            correlation_id=request.correlation_id,
            error="RAG service unavailable",
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
