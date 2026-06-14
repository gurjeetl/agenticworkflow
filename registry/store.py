"""MongoDB-backed persistent store for the agent registry.

MongoDB is chosen because it is the only hard dependency already wired and
index-bootstrapped at app startup (see main.py lifespan + memory/mongo_store.py),
and the repo already uses TTL indexes for expiry. Redis/Postgres are optional in
this project and no-op without env, so they cannot back a service that must
always be available.

Liveness model: a record is "active" iff ``status == "active"`` AND its
``last_heartbeat`` is within ``ttl`` seconds of now. The query enforces freshness
directly (so a just-expired-but-not-yet-swept doc is excluded), while the Mongo
TTL index is the backstop that physically deletes dead docs.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import motor.motor_asyncio

from registry.agent_meta import AgentMeta

COLLECTION = "agent_registry"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RegistryStore:
    def __init__(self) -> None:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "agent_memory")
        self._ttl = int(os.getenv("REGISTRY_TTL_SECONDS", "90"))
        self._client = motor.motor_asyncio.AsyncIOMotorClient(uri)
        self._coll = self._client[db_name][COLLECTION]

    @property
    def ttl_seconds(self) -> int:
        return self._ttl

    async def ensure_indexes(self) -> None:
        await self._coll.create_index("agent_id")
        await self._coll.create_index([("status", 1), ("last_heartbeat", -1)])
        # TTL backstop: Mongo deletes a doc once last_heartbeat is older than TTL.
        await self._coll.create_index("last_heartbeat", expireAfterSeconds=self._ttl)

    # ------------------------------------------------------------------
    async def upsert(self, meta: AgentMeta) -> AgentMeta:
        """Register (or refresh) one agent instance, keyed by instance_id."""
        now = _now()
        meta.last_heartbeat = now
        if meta.registered_at is None:
            meta.registered_at = now
        doc_meta = meta.model_dump(mode="json")
        await self._coll.update_one(
            {"_id": meta.instance_id},
            {
                "$set": {
                    "agent_id": meta.agent_id,
                    "instance_id": meta.instance_id,
                    "version": meta.version,
                    "endpoint": meta.endpoint,
                    "status": meta.status,
                    "meta": doc_meta,
                    "last_heartbeat": now,
                },
                "$setOnInsert": {"registered_at": now},
            },
            upsert=True,
        )
        return meta

    async def heartbeat(self, instance_id: str, status: str | None = None) -> bool:
        """Refresh last_heartbeat. Returns False if the instance is unknown."""
        update: dict = {"last_heartbeat": _now()}
        if status is not None:
            update["status"] = status
            update["meta.status"] = status
        result = await self._coll.update_one(
            {"_id": instance_id}, {"$set": update}
        )
        return result.matched_count > 0

    async def deregister(self, instance_id: str) -> bool:
        result = await self._coll.delete_one({"_id": instance_id})
        return result.deleted_count > 0

    # ------------------------------------------------------------------
    def _fresh_filter(self) -> dict:
        cutoff = _now() - timedelta(seconds=self._ttl)
        return {"status": "active", "last_heartbeat": {"$gte": cutoff}}

    async def list_active(self) -> list[AgentMeta]:
        """All live + active instances, rebuilt from the stored meta blob."""
        cursor = self._coll.find(self._fresh_filter())
        out: list[AgentMeta] = []
        async for doc in cursor:
            out.append(self._to_meta(doc))
        return out

    async def get_agent(self, agent_id: str) -> list[AgentMeta]:
        flt = {**self._fresh_filter(), "agent_id": agent_id}
        cursor = self._coll.find(flt)
        return [self._to_meta(doc) async for doc in cursor]

    @staticmethod
    def _to_meta(doc: dict) -> AgentMeta:
        meta = AgentMeta.model_validate(doc.get("meta", {}))
        # Trust the server-owned liveness fields from the top-level doc.
        meta.last_heartbeat = doc.get("last_heartbeat")
        meta.registered_at = doc.get("registered_at")
        meta.instance_id = doc.get("instance_id")
        return meta

    def close(self) -> None:
        self._client.close()


_store: RegistryStore | None = None


def get_registry_store() -> RegistryStore:
    global _store
    if _store is None:
        _store = RegistryStore()
    return _store
