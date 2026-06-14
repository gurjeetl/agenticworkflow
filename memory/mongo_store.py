import os
from datetime import datetime, timezone

import motor.motor_asyncio
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from pymongo import DESCENDING
from pymongo.errors import OperationFailure

# Hot recent-context cache TTL. Short-term memory is the fast working set for an
# active session; the durable `conversations` collection (no TTL) is the source
# of truth for listing and resuming past conversations.
_SHORT_TERM_TTL_SECONDS = 86400  # 24 hours


def _title_from(messages: list[BaseMessage]) -> str:
    """Derive a conversation title from the first human turn."""
    for m in messages:
        if getattr(m, "type", None) == "human" and getattr(m, "content", None):
            text = str(m.content).strip().replace("\n", " ")
            return (text[:60] + "…") if len(text) > 60 else text
    return "New conversation"


class MongoMemoryStore:
    def __init__(self):
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "agent_memory")
        self._client = motor.motor_asyncio.AsyncIOMotorClient(uri)
        self._db = self._client[db_name]
        self._short_term = self._db["short_term_memory"]
        self._long_term = self._db["long_term_memory"]
        # Durable, never-expiring conversation history powering the conversation
        # list + resume. Survives short-term TTL expiry.
        self._conversations = self._db["conversations"]
        # Structured facts (written by the Synthesizer via the sync FactsStore).
        # Read here so they flow into every agent's prompt; the sync FactsStore
        # owns the schema, scopes (global/session) and the session TTL.
        self._facts = self._db["agent_facts"]

    async def ensure_indexes(self):
        # 24-hour TTL on short-term conversation docs (hot cache only).
        try:
            await self._short_term.create_index("updated_at", expireAfterSeconds=_SHORT_TERM_TTL_SECONDS)
        except OperationFailure as e:
            # Index already exists with a different TTL (e.g. an earlier build).
            # create_index can't change TTL options — update it in place via collMod.
            if getattr(e, "code", None) == 85:  # IndexOptionsConflict
                await self._db.command({
                    "collMod": "short_term_memory",
                    "index": {"keyPattern": {"updated_at": 1}, "expireAfterSeconds": _SHORT_TERM_TTL_SECONDS},
                })
            else:
                raise
        await self._long_term.create_index([("thread_id", 1)])
        # No TTL on conversations — they persist. Sorted by recency for the list.
        await self._conversations.create_index([("updated_at", DESCENDING)])

    async def get_messages(self, thread_id: str) -> list[BaseMessage]:
        """Load prior turns: hot short-term cache first, durable store on a miss.

        On a cache miss (e.g. the 24-hour TTL expired) the durable conversation is
        re-warmed into short-term so the rest of the turn stays fast.
        """
        doc = await self._short_term.find_one({"_id": thread_id})
        if doc and doc.get("messages"):
            return messages_from_dict(doc["messages"])

        conv = await self._conversations.find_one({"_id": thread_id})
        if conv and conv.get("messages"):
            await self._short_term.update_one(
                {"_id": thread_id},
                {"$set": {"messages": conv["messages"], "updated_at": datetime.now(timezone.utc)}},
                upsert=True,
            )
            return messages_from_dict(conv["messages"])
        return []

    async def save_messages(
        self,
        thread_id: str,
        messages: list[BaseMessage],
        session_notes: list[str],
    ):
        dict_msgs = messages_to_dict(messages)
        now = datetime.now(timezone.utc)
        # Hot cache (1-hour TTL).
        await self._short_term.update_one(
            {"_id": thread_id},
            {"$set": {"messages": dict_msgs, "session_notes": session_notes, "updated_at": now}},
            upsert=True,
        )
        # Durable conversation (no TTL). Title is fixed on first write.
        await self._conversations.update_one(
            {"_id": thread_id},
            {
                "$set": {"messages": dict_msgs, "updated_at": now, "message_count": len(dict_msgs)},
                "$setOnInsert": {"created_at": now, "title": _title_from(messages)},
            },
            upsert=True,
        )

    async def list_conversations(self, limit: int = 50) -> list[dict]:
        """Recent conversations for the sidebar: id, title, recency, size."""
        cursor = (
            self._conversations.find({}, {"title": 1, "updated_at": 1, "message_count": 1})
            .sort("updated_at", DESCENDING)
            .limit(limit)
        )
        out: list[dict] = []
        async for d in cursor:
            updated = d.get("updated_at")
            out.append({
                "thread_id": d["_id"],
                "title": d.get("title") or "Conversation",
                "updated_at": updated.isoformat() if updated else None,
                "message_count": d.get("message_count", 0),
            })
        return out

    async def get_conversation(self, thread_id: str) -> list[dict]:
        """Full conversation as simple {role, content} turns for the UI to render."""
        conv = await self._conversations.find_one({"_id": thread_id})
        if not conv or not conv.get("messages"):
            return []
        turns: list[dict] = []
        for m in messages_from_dict(conv["messages"]):
            content = str(getattr(m, "content", "") or "").strip()
            if m.type == "human":
                turns.append({"role": "user", "content": content})
            elif m.type == "ai" and content:
                turns.append({"role": "assistant", "content": content})
        return turns

    async def delete_conversation(self, thread_id: str):
        await self._conversations.delete_one({"_id": thread_id})
        await self._short_term.delete_one({"_id": thread_id})

    async def get_facts(self, thread_id: str) -> dict[str, str]:
        """Facts visible to this thread: all globals plus this thread's session
        facts (session overrides global on a key collision). Read-only — the
        sliding session TTL is bumped by the sync FactsStore on the Planner read
        and Synthesizer write, so this async read stays a pure read (no race)."""
        out: dict[str, str] = {}
        async for doc in self._facts.find({"scope": "global"}):
            out[doc["key"]] = doc["value"]
        async for doc in self._facts.find({"scope": "session", "thread_id": thread_id}):
            out[doc["key"]] = doc["value"]
        return out

    def close(self):
        self._client.close()


_store: MongoMemoryStore | None = None


def get_mongo_store() -> MongoMemoryStore:
    global _store
    if _store is None:
        _store = MongoMemoryStore()
    return _store
