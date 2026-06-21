"""Integration tests for LocalRAGAdapter (pure in-process; no service required)."""
from genie.rag.adapters.local import LocalRAGAdapter
from genie_rag_contracts.retrieval import RetrievalRequest


async def test_local_rag_ingest_and_retrieve() -> None:
    adapter = LocalRAGAdapter()
    await adapter.ingest(
        "ERCOT is the Electric Reliability Council of Texas managing power grid operations.",
        metadata={"document_id": "doc-ercot"},
    )
    request = RetrievalRequest(query="ERCOT power grid", top_k=3)
    response = await adapter.retrieve(request)
    assert response.retrieval_available is True
    assert len(response.results) >= 1
    assert response.results[0].score > 0.0
    assert "ERCOT" in response.results[0].content
    assert response.results[0].document_id == "doc-ercot"


async def test_local_rag_retrieve_empty_returns_empty() -> None:
    adapter = LocalRAGAdapter()
    request = RetrievalRequest(query="nothing here", top_k=5)
    response = await adapter.retrieve(request)
    assert response.retrieval_available is True
    assert response.results == []


async def test_local_rag_retrieves_top_k() -> None:
    adapter = LocalRAGAdapter()
    for i in range(10):
        await adapter.ingest(
            f"Document {i} about energy trading markets and meter data availability.",
            metadata={"document_id": f"doc-{i}"},
        )
    request = RetrievalRequest(query="energy trading meter", top_k=3)
    response = await adapter.retrieve(request)
    assert len(response.results) <= 3


async def test_local_rag_ingest_splits_chunks() -> None:
    adapter = LocalRAGAdapter()
    long_text = " ".join([f"word{i}" for i in range(500)])
    await adapter.ingest(long_text, metadata={"document_id": "big"})
    assert len(adapter._chunks) > 1
