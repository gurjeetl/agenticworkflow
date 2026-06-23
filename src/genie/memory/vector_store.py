"""Optional sync Milvus store: semantic long-term memory (embedded answer vectors
recalled per thread, deduped on write). No-ops when MILVUS_URI/MILVUS_DB_PATH is
unset or pymilvus is missing, so the framework runs without Milvus."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from genie.platform.config import get_settings

_log = logging.getLogger(__name__)

_DEFAULT_COLLECTION = "long_term_memory"
_DEFAULT_EMBED_MODEL = "text-embedding-3-small"
_DEFAULT_DIM = 1536

# Skip re-embedding an answer that's near-identical (cosine >= this) to one already
# stored for the thread. Repeated runs of the same prompt produce slightly-reworded
# answers that would otherwise pile up as near-duplicate vectors and crowd out
# genuinely distinct recall hits. Empirically, reworded versions of one answer sit
# at ~0.94-0.97 cosine while different topics (e.g. outage vs weather) stay <=0.91,
# so 0.93 collapses rewordings without merging genuinely distinct memories.
# Tradeoff: two distinct items with near-identical wording (e.g. two outages whose
# descriptions differ only by id) could merge — acceptable since agent_facts keeps
# the structured, per-entity record; raise this toward 0.97 to be more conservative.
_DEDUP_COSINE = 0.93


class MilvusVectorStore:
    """Milvus-backed semantic long-term memory with OpenAI-compatible embeddings.

    Enabled only when ``MILVUS_URI`` is set and pymilvus is importable; otherwise
    every method no-ops (search → [], add → disabled marker) so the framework
    still runs without standing up Milvus — same pattern as RedisStore.

    Sync on purpose: pymilvus' MilvusClient and ``OpenAIEmbeddings.embed_query``
    are synchronous, and the graph nodes that call this run synchronously.
    """

    def __init__(self) -> None:
        """Resolve the Milvus URI (remote http(s) or local Lite file), guarding the import-time env quirk."""
        # MILVUS_URI may be a remote http(s) endpoint OR a local Milvus Lite file
        # (e.g. ./milvus_local.db — no server/Docker needed). MILVUS_DB_PATH is an
        # explicit alias for the Lite-file case.
        settings = get_settings()
        self._uri = settings.milvus_uri or settings.milvus_db_path
        # pymilvus' ORM reads os.environ['MILVUS_URI'] at import and only accepts
        # http(s) URIs — a local file path there would crash `import pymilvus`
        # process-wide. Move any non-http value out of the env before importing.
        env_uri = os.environ.get("MILVUS_URI")
        if env_uri and not env_uri.lower().startswith("http"):
            os.environ.pop("MILVUS_URI", None)
        self._collection = settings.milvus_collection
        self._embed_model = settings.openai_embed_model
        self._dim = settings.openai_embed_dim
        self._client = None
        self._embeddings = None

        if not self._uri:
            _log.warning("milvus.disabled", extra={"attrs": {"reason": "MILVUS_URI/MILVUS_DB_PATH unset"}})
            return
        try:
            from pymilvus import MilvusClient
        except ImportError:
            _log.warning("milvus.disabled", extra={"attrs": {"reason": "pymilvus not installed"}})
            return
        try:
            self._client = MilvusClient(uri=self._uri, token=get_settings().milvus_token or "")
        except Exception as e:
            _log.warning("milvus.connect_failed", extra={"attrs": {"uri": self._uri, "error": str(e)}})
            self._client = None

    @property
    def enabled(self) -> bool:
        """True only when a Milvus client connected successfully at construction."""
        return self._client is not None

    # ------------------------------------------------------------------
    def _embed(self, text: str) -> list[float] | None:
        """Embed text via the OpenAI-compatible endpoint. None on any failure."""
        try:
            if self._embeddings is None:
                from langchain_openai import OpenAIEmbeddings

                _s = get_settings()
                self._embeddings = OpenAIEmbeddings(
                    model=self._embed_model,
                    api_key=_s.openai_api_key,
                    base_url=_s.openai_base_url or None,
                )
            return self._embeddings.embed_query(text)
        except Exception as e:
            _log.warning("milvus.embed_failed", extra={"attrs": {"error": str(e)}})
            return None

    def ensure_collection(self) -> None:
        """Load the memory collection, creating its schema + cosine index on first
        use. No-op when Milvus is disabled; failures are logged and swallowed."""
        if not self._client:
            return
        try:
            from pymilvus import DataType

            if self._client.has_collection(self._collection):
                # A collection must be loaded into memory before search/query.
                self._client.load_collection(self._collection)
                return
            schema = self._client.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("thread_id", DataType.VARCHAR, max_length=256)
            schema.add_field("content", DataType.VARCHAR, max_length=8192)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self._dim)

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE"
            )
            self._client.create_collection(
                collection_name=self._collection, schema=schema, index_params=index_params
            )
            # Load it so the first search doesn't hit "collection not loaded".
            self._client.load_collection(self._collection)
            _log.info("milvus.collection_ready", extra={"attrs": {"collection": self._collection}})
        except Exception as e:
            _log.warning("milvus.ensure_failed", extra={"attrs": {"error": str(e)}})

    def search(self, thread_id: str, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return up to k semantically-similar past memories for this thread.

        Each hit is {"content": str, "score": float}. Empty list when Milvus or
        embeddings are unavailable.
        """
        if not self._client or not query:
            return []
        vector = self._embed(query)
        if vector is None:
            return []
        try:
            results = self._client.search(
                collection_name=self._collection,
                data=[vector],
                limit=k,
                filter=f'thread_id == "{thread_id}"',
                output_fields=["content"],
                search_params={"metric_type": "COSINE"},
            )
        except Exception as e:
            _log.warning("milvus.search_failed", extra={"attrs": {"error": str(e)}})
            return []
        hits = results[0] if results else []
        out: list[dict[str, Any]] = []
        for h in hits:
            entity = h.get("entity", {}) if isinstance(h, dict) else {}
            out.append({"content": entity.get("content", ""), "score": round(float(h.get("distance", 0.0)), 4)})
        return out

    def _nearest_score(self, thread_id: str, vector: list[float]) -> float | None:
        """Cosine similarity of the closest existing memory for this thread, or None
        if the thread has no memories yet (or Milvus errors)."""
        try:
            results = self._client.search(
                collection_name=self._collection,
                data=[vector],
                limit=1,
                filter=f'thread_id == "{thread_id}"',
                output_fields=[],
                search_params={"metric_type": "COSINE"},
            )
        except Exception as e:
            _log.warning("milvus.dedup_search_failed", extra={"attrs": {"error": str(e)}})
            return None
        hits = results[0] if results else []
        if not hits or not isinstance(hits[0], dict):
            return None
        return float(hits[0].get("distance", 0.0))

    def add(self, thread_id: str, content: str) -> dict[str, Any]:
        """Embed and insert one memory, skipping near-duplicate re-inserts.

        Returns a small op record for the tracer.
        """
        if not self._client:
            return {"enabled": False}
        vector = self._embed(content)
        if vector is None:
            return {"enabled": True, "inserted": False, "reason": "embed_failed"}
        # Dedup-on-write: a prior run's near-identical answer is already stored, so
        # don't add another vector for it (the nearest is from a flushed prior run).
        dup = self._nearest_score(thread_id, vector)
        if dup is not None and dup >= _DEDUP_COSINE:
            return {"enabled": True, "inserted": False, "reason": f"duplicate, cosine {dup:.3f}", "score": dup}
        try:
            self._client.insert(
                collection_name=self._collection,
                data=[{
                    "thread_id": thread_id,
                    "content": content[:8192],
                    "embedding": vector,
                    # created_at kept out of the fixed schema; recorded for the op log only.
                }],
            )
            return {"enabled": True, "inserted": True, "at": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            _log.warning("milvus.insert_failed", extra={"attrs": {"error": str(e)}})
            return {"enabled": True, "inserted": False, "reason": str(e)}

    def close(self) -> None:
        """Close the Milvus client if connected (errors swallowed)."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None


_store: MilvusVectorStore | None = None


def get_vector_store() -> MilvusVectorStore:
    """Return the process-wide MilvusVectorStore singleton, creating it on first use."""
    global _store
    if _store is None:
        _store = MilvusVectorStore()
    return _store
