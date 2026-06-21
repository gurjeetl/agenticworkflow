"""Checkpointer/threading helpers for the compiled graph.

Provides the LangGraph checkpointer that persists state between turns and the
per-thread config that scopes a run to one conversation thread.
"""
from langgraph.checkpoint.memory import MemorySaver


def create_memory() -> MemorySaver:
    """Build the in-memory checkpointer the graph uses to persist per-thread state."""
    return MemorySaver()


def get_thread_config(thread_id: str) -> dict:
    """Wrap a thread id in the ``configurable`` config LangGraph keys checkpoints by."""
    return {"configurable": {"thread_id": thread_id}}
