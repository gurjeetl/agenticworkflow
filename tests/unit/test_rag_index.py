"""Sanity tests for the BM25 DocIndex core (backs both the RAG service and MCP)."""
from genie.rag.index import DocIndex, get_index


def test_docindex_ranks_relevant_chunk_first() -> None:
    chunks = [
        {"source": "a", "text": "the router classifies user intent and picks a route"},
        {"source": "b", "text": "the synthesizer composes a final answer from agent outputs"},
        {"source": "c", "text": "the planner decomposes a goal into a dag of subtasks"},
    ]
    index = DocIndex(chunks)
    hits = index.search("router intent route", k=2)
    assert hits
    assert hits[0]["source"] == "a"
    assert hits[0]["score"] > 0.0


def test_docindex_no_match_returns_empty() -> None:
    index = DocIndex([{"source": "a", "text": "alpha beta gamma"}])
    assert index.search("nonexistentterm", k=4) == []


def test_docindex_empty_corpus_is_safe() -> None:
    assert DocIndex([]).search("anything", k=4) == []


def test_get_index_builds_over_repo_corpus() -> None:
    # The repo ships markdown docs (README, docs/*), so the corpus is non-empty
    # and the moved index still backs the MCP search_docs tool via the shim.
    index = get_index()
    assert index.N > 0
