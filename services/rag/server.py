"""Standalone RAG retrieval/ingestion service.

Independent FastAPI app that serves retrieval over the framework's markdown
corpus (BM25) and accepts ad-hoc ingestion. The platform talks to it through
:class:`~genie.rag.adapters.remote.RemoteRAGAdapter` when ``rag_backend: remote``.

Mirrors the registry service: the wire contracts come from the standalone
``genie_rag_contracts`` package (so this service depends on the package, not on
the platform's internals), while the retrieval core comes from ``genie.rag``.

Run: python -m services.rag.server
Endpoint: http://127.0.0.1:8003
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from genie_rag_contracts.api import INGEST_BASE_PATH, RETRIEVE_PATH
from genie_rag_contracts.ingestion import IngestJobStatus, IngestRequest
from genie_rag_contracts.retrieval import RetrievalRequest, RetrievalResponse

from genie.observability import configure_logging, get_logger
from genie.platform.config import get_settings
from genie.rag.adapters.local import LocalRAGAdapter
from genie.rag.index import get_index

load_dotenv()
configure_logging()
_log = get_logger(__name__)

# One in-process adapter backs the service: seeded from the markdown corpus at
# startup so retrieval answers documentation questions out of the box, and grown
# by the ingestion endpoints at runtime.
_adapter: LocalRAGAdapter | None = None


def _get_adapter() -> LocalRAGAdapter:
    """Return the process-wide adapter, building + corpus-seeding it on first use."""
    global _adapter
    if _adapter is None:
        adapter = LocalRAGAdapter()
        corpus = get_index().chunks
        adapter.seed(
            [
                {
                    "document_id": chunk["source"],
                    "chunk_id": f"{chunk['source']}#{i}",
                    "content": chunk["text"],
                    "metadata": {"source": chunk["source"]},
                }
                for i, chunk in enumerate(corpus)
            ]
        )
        _log.info("rag.corpus_seeded", extra={"attrs": {"chunks": len(corpus)}})
        _adapter = adapter
    return _adapter


def require_auth(authorization: str | None = Header(None)) -> None:
    """Bearer-token gate. No-op when no RAG auth token is configured (local dev)."""
    token = get_settings().rag_service_auth_token
    if not token:
        return
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="invalid rag token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the corpus-seeded index on startup so the first request is fast."""
    adapter = _get_adapter()
    _log.info("rag.ready", extra={"attrs": {"chunks": len(adapter._chunks)}})
    yield


app = FastAPI(title="RAG Service", lifespan=lifespan)


class IngestContentRequest(BaseModel):
    """Client → RAG service: ingest raw ``content`` (no file on disk)."""

    content: str
    document_id: str = ""
    metadata: dict = {}
    correlation_id: str = ""


@app.post(RETRIEVE_PATH, response_model=RetrievalResponse, dependencies=[Depends(require_auth)])
async def retrieve(req: RetrievalRequest) -> RetrievalResponse:
    """Return the top-k chunks for the query, ranked by BM25 over the corpus."""
    return await _get_adapter().retrieve(req)


@app.post(f"{INGEST_BASE_PATH}/content", response_model=IngestJobStatus, dependencies=[Depends(require_auth)])
async def ingest_content(req: IngestContentRequest) -> IngestJobStatus:
    """Ingest raw content into the in-memory index and report job status."""
    doc_id = req.document_id or str(uuid.uuid4())
    metadata = dict(req.metadata)
    metadata["document_id"] = doc_id
    await _get_adapter().ingest(req.content, metadata)
    return IngestJobStatus(
        job_id=str(uuid.uuid4()),
        document_id=doc_id,
        status="completed",
        correlation_id=req.correlation_id,
    )


@app.post(f"{INGEST_BASE_PATH}/file", response_model=IngestJobStatus, dependencies=[Depends(require_auth)])
async def ingest_file(req: IngestRequest) -> IngestJobStatus:
    """Ingest a document by path (read from disk) and report job status."""
    return await _get_adapter().ingest_request(req)


@app.get("/health")
async def health() -> dict:
    """Liveness probe for the RAG service."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("services.rag.server:app", host="127.0.0.1", port=get_settings().rag_service_port)
