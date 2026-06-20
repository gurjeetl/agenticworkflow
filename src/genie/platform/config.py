"""Central platform configuration via pydantic-settings.

A single ``Settings`` object is the platform's source of truth for configuration.
It reads from (in priority order):

1. Process environment variables (the existing unprefixed names — ``OPENAI_MODEL``,
   ``MCP_SERVER_URL``, ``MONGODB_URI``, … — so behavior is unchanged from the old
   scattered ``os.getenv`` calls).
2. A YAML file (``config/default.yaml`` or ``$GENIE_CONFIG_FILE``). Flat keys are
   used only when the matching env var is absent; nested structures
   (``mcp_services``, ``llm_services``) come from YAML because env vars cannot
   express nested dicts.
3. The defaults declared below (identical to the old inline ``os.getenv`` defaults).

Migration note: subsystems are being moved onto ``get_settings()`` incrementally.
Because the fields bind to the same env names the code already used, a module that
still calls ``os.getenv`` and a module that reads ``Settings`` observe the same value.
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
    openai_embed_model: str | None = None
    openai_embed_dim: int | None = None
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
    milvus_collection: str | None = None

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
    router_min_confidence: float = 0.7
    router_intent_classifier: bool = True
    router_intent_model: str | None = None
    router_intent_threshold: float | None = None
    router_intent_min_agents: int | None = None
    router_multi_intent_pattern: str | None = None

    # ── Planner ──────────────────────────────────────────────────────────────
    planner_max_facts: int = 40

    # ── Observability ────────────────────────────────────────────────────────
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "base-agent-framework"
    torch_num_threads: int | None = None
    debug_break: str | None = None

    # ── Misc ─────────────────────────────────────────────────────────────────
    rag_docs_dir: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from a YAML file; env vars win for flat fields."""
        yaml_data: dict[str, Any] = {}
        try:
            import yaml

            with open(path) as fh:
                loaded = yaml.safe_load(fh)
                if isinstance(loaded, dict):
                    yaml_data = loaded
        except (ImportError, OSError):
            pass

        mcp_raw = yaml_data.pop("mcp_services", None)
        llm_raw = yaml_data.pop("llm_services", None)

        instance = cls()  # env + defaults (authoritative for flat fields)

        # Flat YAML values only when the matching env var is absent.
        updates: dict[str, Any] = {
            k: v for k, v in yaml_data.items() if k.upper() not in os.environ
        }
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
    """Return the cached Settings singleton.

    Resolution: ``$GENIE_CONFIG_FILE`` → ``<project_root>/config/default.yaml`` →
    ``config/default.yaml`` in the CWD → env-only.
    """
    global _settings
    if _settings is None:
        with _lock:
            if _settings is None:
                config_file = os.environ.get("GENIE_CONFIG_FILE")
                if config_file is None:
                    # src/genie/platform/config.py → project root is 4 levels up.
                    project_root = Path(__file__).resolve().parents[3]
                    anchor = project_root / "config" / "default.yaml"
                    if anchor.exists():
                        config_file = str(anchor)
                    elif Path("config/default.yaml").exists():
                        config_file = "config/default.yaml"
                _settings = Settings.from_yaml(config_file) if config_file else Settings()
    return _settings


def override_settings(s: Settings) -> None:
    """Replace the cached singleton — intended for test injection."""
    global _settings
    with _lock:
        _settings = s
