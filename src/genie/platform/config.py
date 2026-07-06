"""Central platform configuration via pydantic-settings — env-first (12-factor).

A single ``Settings`` object is the platform's source of truth. Configuration
resolves in this order (highest priority first):

1. Environment variables / ``.env`` (the unprefixed names — ``OPENAI_API_KEY``,
   ``AGENT_PORT``, …). The environment is authoritative: a key set here overrides
   the same key in YAML, so a deploy or container can override committed config
   without editing any file (the 12-factor precedence).
2. ``config/local.yaml`` — gitignored. Secrets (API keys, tokens, DB DSNs) and
   machine-specific overrides. NEVER commit this file.
3. ``config/default.yaml`` — committed. The canonical, non-secret configuration.
   (``$GENIE_CONFIG_FILE`` overrides this path.)
4. The field defaults declared below.

Nested structures (``mcp_services``, ``llm_services``) come only from YAML, since
env vars cannot express nested dicts. All modules read configuration through
``get_settings()`` rather than ``os.getenv`` so the layering applies platform-wide.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Nested config models (loaded from YAML, not env vars) ─────────────────────

class MCPServiceConfig(BaseModel):
    """Connection details for one named MCP server (from YAML mcp_services)."""

    url: str
    transport: str = "sse"
    timeout: float = 30.0
    auth_token: str = ""
    name: str = "default"


class LLMModelConfig(BaseModel):
    """One named OpenAI-compatible LLM backend (from YAML llm_services.models).

    Describes a self-hosted endpoint by host/port/prompting_path; ``base_url``
    derives the URL ChatOpenAI needs (e.g. ``http://host:8033/v1``).
    """

    host: str
    port: int
    model_name: str
    prompting_path: str = "v1"          # URL path segment → base_url ".../{prompting_path}"
    max_token_limit: int | None = None  # model context window (metadata; not max output tokens)
    api_key: str | None = None          # None/blank → "EMPTY" at the use site (open endpoints)
    temperature: float | None = None

    @property
    def base_url(self) -> str:
        """Full prompting endpoint URL (``http://host:port`` plus ``prompting_path`` when set)."""
        path = self.prompting_path.strip("/")
        root = f"http://{self.host}:{self.port}"
        return f"{root}/{path}" if path else root


class LLMServicesConfig(BaseModel):
    """The ``llm_services`` block: named models plus the active default."""

    models: dict[str, LLMModelConfig] = Field(default_factory=dict)
    default: str | None = None

    def active(self) -> LLMModelConfig | None:
        """Return the model selected by ``default``, or None when unset/missing."""
        return self.models.get(self.default) if self.default else None


# ── Top-level Settings ────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """The platform's single configuration object (env > YAML > field defaults).

    Fields bind to the existing unprefixed env-var names. The environment is
    authoritative when both an env var and YAML set a key — see :meth:`from_yaml`
    / :func:`get_settings` for the full layering.
    """

    # env_prefix="" + case_sensitive=False → field `openai_model` binds to env
    # `OPENAI_MODEL`, preserving every existing variable name.
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Runtime environment ──────────────────────────────────────────────────
    # Binds to env var ENVIRONMENT. "development" (the default) enables dev-only
    # conveniences such as the per-agent static port in run_agent; any other value
    # (production, staging, …) disables them so prod uses an explicit port pin or
    # an ephemeral port + discovery. Real deployments set ENVIRONMENT=production.
    environment: str = "development"

    # ── LLM (OpenAI-compatible) ──────────────────────────────────────────────
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    # str (not float) so a blank OPENAI_TEMPERATURE= in .env is treated as "unset"
    # rather than raising a coercion error; coerced to float at the use site.
    openai_temperature: str | None = None
    openai_embed_model: str = "text-embedding-3-small"
    openai_embed_dim: int = 1536
    # Per-component model overrides (None → fall back to openai_model)
    router_model: str | None = None
    planner_model: str | None = None
    synthesizer_model: str | None = None

    # ── MCP (the platform's connection capability) ───────────────────────────
    mcp_server_url: str | None = None
    mcp_server_name: str = "default"
    mcp_transport: str = "sse"
    mcp_auth_token: str = ""
    mcp_timeout: float = 30.0
    # Named MCP servers defined in YAML (key = logical service name).
    mcp_services: dict[str, MCPServiceConfig] = Field(default_factory=dict)

    # Named LLM backends defined in YAML (models + active default).
    llm_services: LLMServicesConfig = Field(default_factory=LLMServicesConfig)

    # ── Persistence ──────────────────────────────────────────────────────────
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "agent_memory"
    redis_url: str | None = None
    milvus_uri: str | None = None
    milvus_db_path: str | None = None
    milvus_token: str | None = None
    milvus_collection: str = "long_term_memory"
    # Optional relational backends — connection strings only; no-op when unset.
    # postgres_dsn: libpq URI, e.g. postgresql://user:pass@host:5432/db
    postgres_dsn: str | None = None
    # sqlserver_dsn: full ODBC connection string (needs "ODBC Driver 18 for SQL Server")
    sqlserver_dsn: str | None = None

    # ── Registry / discovery ─────────────────────────────────────────────────
    registry_url: str = "http://127.0.0.1:8002"
    registry_port: int = 8002
    registry_ttl_seconds: int = 90
    registry_heartbeat_seconds: int = 30
    registry_auth_token: str | None = None
    registry_cache_ttl_s: float = 5.0
    registry_timeout_s: float = 3.0
    registry_serve_stale: bool = True

    # ── Agent service harness ────────────────────────────────────────────────
    agent_host: str = "127.0.0.1"
    # None → no override; the agent's own per-agent default (passed to run_agent)
    # applies, falling back to an OS-assigned ephemeral port. Set AGENT_PORT (env,
    # top priority) or agent_port (YAML) to pin a specific port.
    agent_port: int | None = None
    agent_advertise_host: str | None = None
    agent_advertise_port: int | None = None
    agent_invoke_token: str | None = None

    # ── A2A protocol / Agent Card ────────────────────────────────────────────
    # Advertised A2A protocol version. Defaults to the code's PROTOCOL_VERSION;
    # lets ops pin a version without a code change. None → use PROTOCOL_VERSION.
    a2a_protocol_version: str | None = None
    # Optional AgentProvider fields surfaced on every served Agent Card.
    agent_provider_organization: str | None = None
    agent_provider_url: str | None = None

    # ── Security guard ───────────────────────────────────────────────────────
    # Master switch for the input/output content guard. When False the pipeline
    # omits BOTH guard nodes and never loads the llm-guard models — so the app
    # runs UNPROTECTED. Left ON by default (fail-safe). Set LLM_GUARD_ENABLED=0
    # to disable (e.g. local dev, tests, or when content is guarded upstream).
    # The flags below only take effect when the guard is enabled.
    llm_guard_enabled: bool = True
    llm_guard_pii: bool = True
    llm_guard_parallel: bool = True
    llm_guard_use_onnx: bool = False
    llm_guard_onnx_quantized: bool = False
    llm_guard_fail_open: bool = False
    llm_guard_ban_topics: str | None = None
    llm_guard_injection_patterns: str | None = None

    # ── Router ───────────────────────────────────────────────────────────────
    # Master switch for the Router triage node. When False the pipeline skips the
    # Router entirely and every prompt goes straight to the full Planner pipeline
    # (no fast-path / chitchat shortcut). Set ROUTER_ENABLED=0 to disable.
    router_enabled: bool = False
    router_min_confidence: float = 0.7
    # Backend for the Router's multi-intent detection (which routes 2+-agent
    # prompts straight to the Planner):
    #   "embedding" — local sentence-transformers model (router_intent_model).
    #                 Fast and offline ONCE cached, but the first load downloads
    #                 from HuggingFace. Use where the model is reachable/pre-cached.
    #   "llm"       — no local model: the Router's own LLM route call classifies
    #                 intent (a multi-intent prompt is routed to "plan"). Use where
    #                 HuggingFace is unreachable — adds NO extra LLM call, it just
    #                 relies on the route decision the Router already makes.
    # Set ROUTER_INTENT_BACKEND=llm to avoid loading the local model entirely.
    router_intent_backend: str = "embedding"
    # Deprecated alias kept for back-compat: router_intent_classifier=False forces
    # the "llm" backend (the old "regex-only, let the LLM decide" behavior). Prefer
    # router_intent_backend. True leaves router_intent_backend in control.
    router_intent_classifier: bool = True
    router_intent_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    router_intent_threshold: float = 0.30
    router_intent_min_agents: int = 2
    router_multi_intent_pattern: str | None = None

    # ── Planner ──────────────────────────────────────────────────────────────
    planner_max_facts: int = 40

    # ── Observability ────────────────────────────────────────────────────────
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "base-agent-framework"
    torch_num_threads: int | None = None
    debug_break: str | None = None

    # ── RAG (retrieval) ──────────────────────────────────────────────────────
    # Override the markdown corpus root indexed by genie.rag.index (defaults to
    # the repo root when unset).
    rag_docs_dir: str | None = None
    # Adapter selection: "local" (in-process BM25) or "remote" (standalone service).
    rag_backend: str = "local"
    rag_service_url: str = "http://127.0.0.1:8003"
    rag_service_port: int = 8003
    rag_service_timeout_s: float = 3.0
    rag_service_auth_token: str | None = None

    @property
    def is_development(self) -> bool:
        """True when running in development mode (enables dev-only conveniences).

        Driven by the ``environment`` setting / ``ENVIRONMENT`` env var; any value
        other than "development" (production, staging, …) returns False.
        """
        return self.environment.strip().lower() == "development"

    @classmethod
    def from_yaml(cls, *paths: str | Path) -> "Settings":
        """Build Settings layering the environment over YAML over field defaults.

        The **environment is authoritative**: a key set via an env var (or
        ``.env``) overrides the same key in YAML — the 12-factor precedence, so a
        deploy/container can override committed config without editing files.
        Multiple YAML files are layered in order (later files win), so
        ``from_yaml(default_yaml, local_yaml)`` lets a gitignored ``local.yaml``
        override the committed ``default.yaml``; YAML in turn overrides the field
        defaults. Nested ``mcp_services`` / ``llm_services`` come only from YAML
        (env vars cannot express nested dicts).
        """
        merged: dict[str, Any] = {}
        for path in paths:
            if not path:
                continue
            try:
                import yaml

                with open(path) as fh:
                    loaded = yaml.safe_load(fh)
                    if isinstance(loaded, dict):
                        merged.update(loaded)  # later file overrides earlier
            except (ImportError, OSError):
                pass

        mcp_raw = merged.pop("mcp_services", None)
        llm_raw = merged.pop("llm_services", None)

        instance = cls()  # env + .env + field defaults
        # Keys the environment explicitly set (via env var or .env). pydantic only
        # records sourced fields here — defaults are absent — so the environment
        # wins: YAML fills only the keys the environment did NOT set.
        env_set = set(instance.model_fields_set)

        updates: dict[str, Any] = {k: v for k, v in merged.items() if k not in env_set}
        # Nested structures can only come from YAML (env can't express them).
        if mcp_raw and isinstance(mcp_raw, dict):
            updates["mcp_services"] = {
                name: MCPServiceConfig.model_validate(svc) for name, svc in mcp_raw.items()
            }
        if llm_raw and isinstance(llm_raw, dict):
            updates["llm_services"] = LLMServicesConfig.model_validate(llm_raw)
        return instance.model_copy(update=updates)


# ── Singleton cache ───────────────────────────────────────────────────────────
_settings: Settings | None = None
_lock = threading.Lock()


def get_settings() -> Settings:
    """Return the cached Settings singleton (env > YAML > field defaults).

    YAML layering (later wins): ``config/default.yaml`` (committed, the canonical
    config) then ``config/local.yaml`` (gitignored — secrets and machine-specific
    overrides). Environment variables / ``.env`` are authoritative over both: an
    env var overrides the same key in YAML (the 12-factor precedence), so a
    deploy/container can override committed config without editing files.
    ``$GENIE_CONFIG_FILE`` overrides the default.yaml location.
    """
    global _settings
    if _settings is None:
        with _lock:
            if _settings is None:
                # src/genie/platform/config.py → project root is 4 levels up.
                project_root = Path(__file__).resolve().parents[3]
                config_dir = project_root / "config"

                base = os.environ.get("GENIE_CONFIG_FILE")
                if base is None:
                    anchor = config_dir / "default.yaml"
                    base = str(anchor) if anchor.exists() else (
                        "config/default.yaml" if Path("config/default.yaml").exists() else None
                    )

                local = config_dir / "local.yaml"
                local_path = str(local) if local.exists() else (
                    "config/local.yaml" if Path("config/local.yaml").exists() else None
                )

                paths = [p for p in (base, local_path) if p]
                _settings = Settings.from_yaml(*paths) if paths else Settings()
    return _settings


def override_settings(s: Settings) -> None:
    """Replace the cached singleton — intended for test injection."""
    global _settings
    with _lock:
        _settings = s
