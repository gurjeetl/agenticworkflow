"""Retrieval-side wire models for the RAG service boundary."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RetrievalRequest(BaseModel):
    """Client → RAG service: retrieve the top-k chunks for ``query``."""

    query: str
    top_k: int = 5
    filters: dict[str, Any] = {}
    correlation_id: str = ""


class RetrievalResult(BaseModel):
    """One retrieved chunk with its provenance and relevance score."""

    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = {}


class RetrievalResponse(BaseModel):
    """RAG service → client: the ranked results for one retrieval request.

    ``retrieval_available`` is False when a client (e.g. the remote adapter)
    degraded to an empty response after a service/network error, so callers can
    distinguish "no matches" from "RAG was unreachable".
    """

    results: list[RetrievalResult]
    query: str
    correlation_id: str = ""
    retrieval_available: bool = True
