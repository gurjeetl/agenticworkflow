"""Conversation history endpoints for the sidebar (list / resume / delete)."""
from fastapi import APIRouter

from genie.memory.mongo_store import get_mongo_store

router = APIRouter()


@router.get("/conversations")
async def list_conversations(limit: int = 50):
    """List past conversations (durable) for the sidebar, most recent first."""
    store = get_mongo_store()
    return {"conversations": await store.list_conversations(limit=limit)}


@router.get("/conversations/{thread_id}")
async def get_conversation(thread_id: str):
    """Full history of one conversation as {role, content} turns, for resuming."""
    store = get_mongo_store()
    return {"thread_id": thread_id, "turns": await store.get_conversation(thread_id)}


@router.delete("/conversations/{thread_id}")
async def delete_conversation(thread_id: str):
    """Delete one conversation's durable history (sidebar removal)."""
    store = get_mongo_store()
    await store.delete_conversation(thread_id)
    return {"deleted": thread_id}
