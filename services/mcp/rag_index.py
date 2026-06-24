"""Backward-compatible shim — the RAG index now lives in :mod:`genie.rag.index`.

The BM25 ``DocIndex`` / ``get_index`` logic moved into the platform package so it
can back both the standalone RAG service ([services/rag/server.py]) and the
in-process MCP ``search_docs`` tool ([services/mcp/genie_mcp_server.py]). This module
re-exports the same names so existing imports (``from services.mcp.rag_index
import get_index``) keep working unchanged.
"""
from __future__ import annotations

from genie.rag.index import DocIndex, get_index

__all__ = ["DocIndex", "get_index"]
