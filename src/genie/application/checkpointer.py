"""Checkpointer/threading helpers for the compiled graph.

Provides the LangGraph checkpointer that persists state between turns and the
per-thread config that scopes a run to one conversation thread.

Two modes:

* **Sync-only platform** (``kafka_enabled=False``, the default): the in-memory
  ``MemorySaver`` — exactly the old behavior, no new infrastructure required.
* **Async A2A mode** (``kafka_enabled=True``): a **MongoDB-backed saver**. A bus
  task suspends the run via ``interrupt()`` and a *different* process/loop (the
  gateway reply-router) resumes it later — possibly after a gateway restart —
  so the checkpoint must live outside process memory. Mongo is the platform's
  one always-available store, so it hosts the checkpoints too.
"""
from genie.observability import get_logger
from genie.platform.config import get_settings

_log = get_logger(__name__)


def create_memory():
    """Build the graph checkpointer: Mongo-backed in async mode, else in-memory."""
    settings = get_settings()
    if settings.kafka_enabled:
        from langgraph.checkpoint.mongodb import MongoDBSaver

        from genie.platform.mongo import get_sync_mongo_client

        _log.info("checkpointer.mongodb", extra={"attrs": {"db": settings.mongodb_db}})
        return MongoDBSaver(get_sync_mongo_client(), db_name=settings.mongodb_db)

    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


def get_thread_config(thread_id: str) -> dict:
    """Wrap a thread id in the ``configurable`` config LangGraph keys checkpoints by."""
    return {"configurable": {"thread_id": thread_id}}
