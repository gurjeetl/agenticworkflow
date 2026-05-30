import os
from datetime import datetime, timezone

import motor.motor_asyncio
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


class MongoMemoryStore:
    def __init__(self):
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "agent_memory")
        self._client = motor.motor_asyncio.AsyncIOMotorClient(uri)
        db = self._client[db_name]
        self._short_term = db["short_term_memory"]
        self._long_term = db["long_term_memory"]

    async def ensure_indexes(self):
        # 24-hour TTL on short-term conversation docs
        await self._short_term.create_index("updated_at", expireAfterSeconds=86400)
        await self._long_term.create_index([("thread_id", 1)])

    async def get_messages(self, thread_id: str) -> list[BaseMessage]:
        doc = await self._short_term.find_one({"_id": thread_id})
        if not doc or not doc.get("messages"):
            return []
        return messages_from_dict(doc["messages"])

    async def save_messages(
        self,
        thread_id: str,
        messages: list[BaseMessage],
        session_notes: list[str],
    ):
        await self._short_term.update_one(
            {"_id": thread_id},
            {
                "$set": {
                    "messages": messages_to_dict(messages),
                    "session_notes": session_notes,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )

    async def upsert_fact(self, thread_id: str, key: str, value: str):
        doc_id = f"{thread_id}__{key}"
        await self._long_term.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    "thread_id": thread_id,
                    "key": key,
                    "value": value,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )

    async def get_facts(self, thread_id: str) -> dict[str, str]:
        cursor = self._long_term.find({"thread_id": thread_id})
        return {doc["key"]: doc["value"] async for doc in cursor}

    async def delete_fact(self, thread_id: str, key: str):
        await self._long_term.delete_one({"_id": f"{thread_id}__{key}"})

    def close(self):
        self._client.close()


_store: MongoMemoryStore | None = None


def get_mongo_store() -> MongoMemoryStore:
    global _store
    if _store is None:
        _store = MongoMemoryStore()
    return _store
