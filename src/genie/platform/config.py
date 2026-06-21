"""Central platform configuration via pydantic-settings — YAML-first.

A single ``Settings`` object is the platform's source of truth. Configuration is
driven by YAML; resolution order (highest priority first):

1. ``config/local.yaml`` — gitignored. Secrets (API keys, tokens, DB DSNs) and
   machine-specific overrides. NEVER commit this file.
2. ``config/default.yaml`` — committed. The canonical, non-secret configuration.
   (``$GENIE_CONFIG_FILE`` overrides this path.)
3. Environment variables (the unprefixed names — ``OPENAI_API_KEY``, ``MONGODB_URI``,
   …). Only used for keys NOT set in any YAML — a fallback, no longer the primary
   path. Useful for injecting a secret without writing it to a file.
4. The field defaults declared below.

YAML is authoritative: a key set in YAML overrides the same environment variable.
Nested structures (``mcp_services``, ``llm_services``) come only from YAML, since
env vars cannot express nested dicts. All modules read configuration through
``get_settings()`` rather than ``os.getenv`` so YAML drives the whole platform.
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
    """Configuration for one named LLM backend (from YAML llm_services)."""

    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None


# ── Top-level Settings ────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """The platform's single configuration object (env > YAML > field defaults).

    Fields bind to the existing unprefixed env-var names so a module reading
    ``Settings`` and one still calling ``os.getenv`` observe identical values.
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

    # Named LLM backends defined in YAML (key = logical name).
    llm_services: dict[str, LLMModelConfig] = Field(default_factory=dict)

    # ── Persistence ──────────────────────────────────────────────────────────
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "agent_memory"
    redis_url: str | None = None
    milvus_uri: str | None = None
    milvus_db_path: str | None = None
    milvus_token: str | None = None
    milvus_collection: str = "long_term_memory"

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
    agent_port: int = 8010
    agent_advertise_host: str | None = None
    agent_advertise_port: int | None = None
    agent_invoke_token: str | None = None

    # ── Security guard ───────────────────────────────────────────────────────
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

    @classmethod
    def from_yaml(cls, *paths: str | Path) -> "Settings":
        """Build Settings from one or more YAML files layered over env + defaults.

        YAML is **authoritative**: a value set in YAML overrides the same key from
        an environment variable. Multiple files are layered in order (later files
        win), so ``from_yaml(default_yaml, local_yaml)`` lets a gitignored
        ``local.yaml`` override the committed ``default.yaml``. Keys absent from
        every YAML fall back to the env var (useful for secrets) then the field
        default. Nested ``mcp_services`` / ``llm_services`` come only from YAML.
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

        instance = cls()  # env + field defaults — the fallback for keys not in YAML

        # YAML flat values are authoritative — they override env for any key they set.
        updates: dict[str, Any] = dict(merged)
        if mcp_raw and isinstance(mcp_raw, dict):
            updates["mcp_services"] = {
                name: MCPServiceConfig.model_validate(svc) for name, svc in mcp_raw.items()
            }
        if llm_raw and isinstance(llm_raw, dict):
            updates["llm_services"] = {
                name: LLMModelConfig.model_validate(svc) for name, svc in llm_raw.items()
            }
        return instance.model_copy(update=updates)


# ── Singleton cache ───────────────────────────────────────────────────────────
_settings: Settings | None = None
_lock = threading.Lock()


def get_settings() -> Settings:
    """Return the cached Settings singleton, sourced primarily from YAML.

    Layering (later wins): ``config/default.yaml`` (committed, the canonical
    config) then ``config/local.yaml`` (gitignored — secrets and machine-specific
    overrides). Both are authoritative over environment variables; env vars only
    fill keys absent from every YAML (handy for secrets you'd rather not put in a
    file). ``$GENIE_CONFIG_FILE`` overrides the default.yaml location.
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
