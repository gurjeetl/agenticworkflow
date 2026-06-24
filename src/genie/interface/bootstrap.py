"""Application factory for the genie platform gateway.

``create_app()`` builds the agent-agnostic FastAPI app: it configures logging and
MLflow, owns the lifespan (store init + mandatory guard warm-up), wires the REST
routers, and mounts the static frontend. Concrete agents are NOT injected here —
they run as their own services and are discovered at runtime through the Registry
(see ``genie.registry``), so this factory takes no agent providers.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from genie.observability import configure_logging, get_logger, init_mlflow
from genie.platform.config import Settings, get_settings

# Repo root resolved from this file (src/genie/interface/bootstrap.py → 4 levels up),
# so the frontend mount works regardless of the CWD uvicorn starts in.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"


def _make_lifespan(settings: Settings):
    """Build the FastAPI lifespan: bootstrap stores + guards on startup, close on exit."""
    _log = get_logger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Init/index stores and warm the mandatory guard + intent models, then teardown."""
        # Optional cap on PyTorch intra-op threads for the local models (LLM Guard +
        # router intent classifier). Each model otherwise grabs all cores, so when two
        # requests overlap their inferences oversubscribe the CPU and BOTH slow down.
        # Capping trades a little single-request latency for less thrash under
        # concurrency. Unset = PyTorch default (no change); tune per box.
        if settings.torch_num_threads:
            try:
                import torch

                torch.set_num_threads(int(settings.torch_num_threads))
                _log.info("torch.num_threads_set", extra={"attrs": {"threads": int(settings.torch_num_threads)}})
            except Exception as e:  # torch missing / bad value — never block startup
                _log.warning("torch.num_threads_failed", extra={"attrs": {"error": str(e)}})

        from genie.memory.commit_store import get_commit_store
        from genie.memory.facts_store import get_facts_store
        from genie.memory.mongo_store import get_mongo_store
        from genie.memory.redis_store import get_redis_store
        from genie.memory.vector_store import get_vector_store
        from genie.security import get_llm_guard

        store = get_mongo_store()
        await store.ensure_indexes()
        _log.info("mongodb.indexes_ensured")

        commits = get_commit_store()
        commits.ensure_indexes()
        _log.info("commit_store.ready", extra={"attrs": {"enabled": commits.enabled}})

        # Shares the commit store's pymongo client, so no separate close() below.
        facts = get_facts_store()
        facts.ensure_indexes()
        _log.info("facts_store.ready", extra={"attrs": {"enabled": facts.enabled}})

        vectors = get_vector_store()
        vectors.ensure_collection()
        _log.info("milvus.ready", extra={"attrs": {"enabled": vectors.enabled}})

        redis = get_redis_store()
        _log.info("redis.ready", extra={"attrs": {"enabled": redis.enabled}})

        # Optional relational backends (Postgres / SQL Server). Best-effort: when a
        # DSN is configured, run the health-check so a bad connection surfaces in the
        # logs at startup; never block boot if it fails.
        if settings.postgres_dsn:
            try:
                from genie.platform.postgres import postgres_healthcheck

                _log.info("postgres.ready", extra={"attrs": {"ok": postgres_healthcheck()}})
            except Exception as e:
                _log.warning("postgres.healthcheck_failed", extra={"attrs": {"error": str(e)}})
        if settings.sqlserver_dsn:
            try:
                from genie.platform.sqlserver import sqlserver_healthcheck

                _log.info("sqlserver.ready", extra={"attrs": {"ok": sqlserver_healthcheck()}})
            except Exception as e:
                _log.warning("sqlserver.healthcheck_failed", extra={"attrs": {"error": str(e)}})

        # Content guard (ON by default; LLM_GUARD_ENABLED=0 to disable). When on,
        # constructing it here loads the local models, so a missing dependency or
        # un-loadable model aborts startup (fail-closed) rather than letting the
        # pipeline run unprotected. When off, the graph omits the guard nodes too,
        # so we skip the load entirely.
        if settings.llm_guard_enabled:
            get_llm_guard().warm()  # load AND warm the kernels so the first request pays neither
            _log.info("llm_guard.ready")
        else:
            _log.warning("llm_guard.disabled", extra={"attrs": {"reason": "LLM_GUARD_ENABLED=0"}})

        # Warm the Router's local multi-intent classifier so the first request doesn't
        # pay the model load. Best-effort: it fails open if the model can't load. With
        # the "llm" intent backend this is a no-op (no local model / no HuggingFace).
        from genie.application.nodes._router_intent import get_intent_classifier

        classifier = get_intent_classifier()
        classifier.warm()
        _log.info("router_intent_classifier.ready", extra={"attrs": {"backend": classifier.backend}})

        yield

        from genie.platform.db import close_all_connections

        await close_all_connections()

    return lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the gateway FastAPI app."""
    configure_logging()
    init_mlflow()

    settings = settings or get_settings()

    app = FastAPI(lifespan=_make_lifespan(settings))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Build the graph eagerly so a construction error surfaces at startup, not on
    # the first request. Routers fetch the same singleton via get_graph().
    from genie.application.graph import get_graph

    get_graph()

    from genie.interface.routers import chat, conversations, health, registry, state

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(state.router)
    app.include_router(registry.router)
    app.include_router(conversations.router)

    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
    return app
