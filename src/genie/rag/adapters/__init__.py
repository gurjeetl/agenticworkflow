"""RAG adapters — local in-process and remote HTTP implementations."""
from genie.rag.adapters.local import LocalRAGAdapter
from genie.rag.adapters.remote import RemoteRAGAdapter

__all__ = ["LocalRAGAdapter", "RemoteRAGAdapter"]
