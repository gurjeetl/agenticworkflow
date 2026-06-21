"""Shared contracts for the genie RAG service boundary."""
from .api import API_VERSION, CORRELATION_ID_HEADER, INGEST_BASE_PATH, RETRIEVE_PATH
from .ingestion import IngestJobStatus, IngestRequest
from .retrieval import RetrievalRequest, RetrievalResponse, RetrievalResult

__all__ = [
    "API_VERSION",
    "CORRELATION_ID_HEADER",
    "INGEST_BASE_PATH",
    "RETRIEVE_PATH",
    "IngestRequest",
    "IngestJobStatus",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievalResponse",
]
