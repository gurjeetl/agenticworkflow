"""Ingestion-side wire models for the RAG service boundary."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Client → RAG service: ingest one document identified by ``document_path``."""

    document_path: str
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = {}
    correlation_id: str = ""


class IngestJobStatus(BaseModel):
    """RAG service → client: lifecycle status of one ingestion job."""

    job_id: str
    document_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    correlation_id: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    chunk_count: Optional[int] = None
