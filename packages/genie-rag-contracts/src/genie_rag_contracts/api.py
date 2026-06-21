"""Stable API constants shared by the RAG service and its clients.

These pin the wire surface (version, correlation header, route paths) so the
standalone RAG service and the platform-side adapters agree on URLs and headers
without importing each other's code.
"""
API_VERSION = "v1"
CORRELATION_ID_HEADER = "X-Correlation-ID"
INGEST_BASE_PATH = f"/{API_VERSION}/ingest"
RETRIEVE_PATH = f"/{API_VERSION}/retrieve"
