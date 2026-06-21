"""Unit tests for the genie_rag_contracts wire models (defaults + round-trips)."""
from genie_rag_contracts.api import API_VERSION, RETRIEVE_PATH
from genie_rag_contracts.ingestion import IngestJobStatus, IngestRequest
from genie_rag_contracts.retrieval import RetrievalRequest, RetrievalResponse, RetrievalResult


def test_retrieval_request_defaults() -> None:
    req = RetrievalRequest(query="hello")
    assert req.top_k == 5
    assert req.filters == {}
    assert req.correlation_id == ""


def test_retrieval_response_defaults() -> None:
    resp = RetrievalResponse(results=[], query="hello")
    assert resp.retrieval_available is True
    assert resp.results == []


def test_retrieval_response_round_trip() -> None:
    resp = RetrievalResponse(
        results=[RetrievalResult(document_id="d1", chunk_id="d1:0", content="x", score=1.0)],
        query="hello",
        correlation_id="c1",
    )
    rebuilt = RetrievalResponse.model_validate(resp.model_dump())
    assert rebuilt == resp
    assert rebuilt.results[0].chunk_id == "d1:0"


def test_ingest_request_auto_document_id() -> None:
    req = IngestRequest(document_path="/tmp/x.md")
    assert req.document_id  # auto-generated uuid
    rebuilt = IngestRequest.model_validate(req.model_dump())
    assert rebuilt.document_id == req.document_id


def test_ingest_job_status_round_trip() -> None:
    status = IngestJobStatus(job_id="j1", document_id="d1", status="completed", chunk_count=3)
    rebuilt = IngestJobStatus.model_validate(status.model_dump())
    assert rebuilt.status == "completed"
    assert rebuilt.chunk_count == 3


def test_api_constants() -> None:
    assert API_VERSION == "v1"
    assert RETRIEVE_PATH == "/v1/retrieve"
