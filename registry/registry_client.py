"""HTTP client the Planner and Executor use to discover agents.

Replaces direct access to the old in-process ``AGENT_REGISTRY`` dict. Returns the
same :class:`AgentMeta` objects the rest of the code already expects, so
downstream logic (menu rendering, ``validate_args``, ``version``/``sla_ms``
reads, the new ``endpoint`` lookup) is unchanged.

A short in-process TTL cache absorbs the fact that a single Planner run renders
the capability menu once and validates N subtasks — all served from one HTTP
fetch per window. A lock guards the cache because the Executor reaches the client
from worker threads; the network call itself happens outside the lock.
"""
from __future__ import annotations

import os
import threading
import time

import httpx

from observability import get_logger
from registry.agent_meta import AgentMeta

_log = get_logger(__name__)


class RegistryUnavailable(RuntimeError):
    """Registry Service unreachable or returned a malformed response."""


class RegistryClient:
    def __init__(
        self,
        base_url: str | None = None,
        cache_ttl_s: float | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._base_url = (base_url or os.getenv("REGISTRY_URL", "http://127.0.0.1:8002")).rstrip("/")
        self._cache_ttl = float(cache_ttl_s if cache_ttl_s is not None else os.getenv("REGISTRY_CACHE_TTL_S", "5"))
        self._timeout = float(timeout_s if timeout_s is not None else os.getenv("REGISTRY_TIMEOUT_S", "3"))
        self._serve_stale = os.getenv("REGISTRY_SERVE_STALE", "1") == "1"
        token = os.getenv("REGISTRY_AUTH_TOKEN")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._lock = threading.Lock()
        self._cache: list[AgentMeta] | None = None
        self._cache_at = 0.0
        self._client = httpx.Client(timeout=self._timeout)

    # ------------------------------------------------------------------
    def list_active(self, *, force_refresh: bool = False) -> list[AgentMeta]:
        with self._lock:
            fresh = self._cache is not None and (time.monotonic() - self._cache_at) < self._cache_ttl
            if fresh and not force_refresh:
                return self._cache
        try:
            agents = self._fetch_agents()  # network outside the lock
        except RegistryUnavailable:
            with self._lock:
                if self._serve_stale and self._cache is not None:
                    _log.warning("registry_client.serving_stale")
                    return self._cache
            raise
        with self._lock:
            self._cache, self._cache_at = agents, time.monotonic()
        return agents

    def get(self, agent_id: str) -> AgentMeta | None:
        return next((m for m in self.list_active() if m.agent_id == agent_id), None)

    def invalidate(self) -> None:
        with self._lock:
            self._cache, self._cache_at = None, 0.0

    # ------------------------------------------------------------------
    def _fetch_agents(self) -> list[AgentMeta]:
        try:
            resp = self._client.get(f"{self._base_url}/agents", headers=self._headers)
            resp.raise_for_status()
            raw = resp.json().get("agents", [])
        except (httpx.HTTPError, ValueError) as e:
            _log.warning("registry_client.fetch_failed", extra={"attrs": {"error": str(e)}})
            raise RegistryUnavailable(str(e)) from e
        out: list[AgentMeta] = []
        for rec in raw:
            try:
                out.append(AgentMeta.model_validate(rec))
            except Exception as e:  # tolerate one bad record without failing discovery
                _log.warning("registry_client.bad_record", extra={"attrs": {"error": str(e)}})
        return [m for m in out if m.status == "active"]


_default_client: RegistryClient | None = None


def get_registry_client() -> RegistryClient:
    global _default_client
    if _default_client is None:
        _default_client = RegistryClient()
    return _default_client
