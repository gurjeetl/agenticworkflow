"""LocalRAGAdapter — in-process retrieval for zero-dependency / CI mode.

Satisfies the ``RetrievalService`` + ``IngestionService`` protocols without a
running RAG service. Ingested content is chunked on word boundaries and scored
with the same BM25 :class:`~genie.rag.index.DocIndex` the standalone service uses,
so local and remote retrieval rank consistently.
"""
from __future__ import annotations

import uuid
from typing import Any

from genie_rag_contracts.ingestion import IngestJobStatus, IngestRequest
from genie_rag_contracts.retrieval import RetrievalRequest, RetrievalResponse, RetrievalResult

from genie.observability import get_logger
from genie.rag.index import DocIndex

_log = get_logger(__name__)


def _split_chunks(text: str, max_chars: int = 512) -> list[str]:
    """Split text into chunks of at most ``max_chars`` on word boundaries."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if length + len(word) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks or [""]


class LocalRAGAdapter:
    """In-process RAG adapter backed by an in-memory BM25 index.

    Suitable for local development and CI without running the RAG service.
    ``ingest`` appends chunks; ``retrieve`` (re)builds a :class:`DocIndex` over
    the accumulated chunks and returns the top-k by BM25 score.
    """

    def __init__(self) -> None:
        # One entry per chunk: {document_id, chunk_id, content, metadata}.
        self._chunks: list[dict[str, Any]] = []
        self._by_chunk: dict[str, dict[str, Any]] = {}
        self._docindex: DocIndex | None = None  # lazily (re)built on retrieve

    def seed(self, chunks: list[dict[str, Any]]) -> None:
        """Bulk-load pre-chunked entries (``document_id``/``chunk_id``/``content``/``metadata``).

        Used by the standalone service to preload the markdown corpus without
        re-chunking. Invalidates the cached index so the next retrieve rebuilds it.
        """
        for entry in chunks:
            self._chunks.append(entry)
            self._by_chunk[entry["chunk_id"]] = entry
        self._docindex = None

    # ── IngestionService ──────────────────────────────────────────────────────

    async def ingest(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Chunk ``content`` and append it to the in-memory index."""
        meta = metadata or {}
        doc_id = meta.get("document_id", str(uuid.uuid4()))
        for i, chunk in enumerate(_split_chunks(content)):
            chunk_id = f"{doc_id}:{i}"
            entry = {
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "content": chunk,
                "metadata": meta,
            }
            self._chunks.append(entry)
            self._by_chunk[chunk_id] = entry
        self._docindex = None  # invalidate cached index
        _log.debug("local_rag.ingested", extra={"attrs": {"doc_id": doc_id, "chunks": len(self._chunks)}})

    async def ingest_request(self, request: IngestRequest) -> IngestJobStatus:
        """Ingest a document by path (reads file content from disk)."""
        job_id = str(uuid.uuid4())
        try:
            with open(request.document_path, encoding="utf-8") as fh:
                content = fh.read()
            metadata = dict(request.metadata)
            metadata["document_id"] = request.document_id
            await self.ingest(content, metadata)
            return IngestJobStatus(
                job_id=job_id,
                document_id=request.document_id,
                status="completed",
                correlation_id=request.correlation_id,
                chunk_count=len(_split_chunks(content)),
            )
        except Exception as exc:
            return IngestJobStatus(
                job_id=job_id,
                document_id=request.document_id,
                status="failed",
                correlation_id=request.correlation_id,
                error=str(exc),
            )

    # ── RetrievalService ──────────────────────────────────────────────────────

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Return the top-k chunks ranked by BM25 over the ingested corpus."""
        index = self._ensure_index()
        hits = index.search(request.query, k=request.top_k)
        results: list[RetrievalResult] = []
        for hit in hits:
            entry = self._by_chunk.get(hit["source"], {})
            results.append(
                RetrievalResult(
                    document_id=entry.get("document_id", hit["source"]),
                    chunk_id=hit["source"],
                    content=hit["text"],
                    score=hit["score"],
                    metadata=entry.get("metadata", {}),
                )
            )
        _log.debug(
            "local_rag.retrieved",
            extra={"attrs": {"query": request.query[:80], "result_count": len(results)}},
        )
        return RetrievalResponse(
            results=results,
            query=request.query,
            correlation_id=request.correlation_id,
            retrieval_available=True,
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _ensure_index(self) -> DocIndex:
        """Build (and cache) a BM25 index over the ingested chunks; ``source``=chunk_id."""
        if self._docindex is None:
            self._docindex = DocIndex(
                [{"source": e["chunk_id"], "text": e["content"]} for e in self._chunks]
            )
        return self._docindex
