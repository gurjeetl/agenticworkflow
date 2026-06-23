"""Registry surface: agent-metadata models and the client for discovering live agents."""
from genie.registry.agent_meta import AgentMeta, FieldSpec, Skill
from genie.registry.registry_client import (
    RegistryClient,
    RegistryUnavailable,
    get_registry_client,
)

__all__ = [
    "AgentMeta",
    "FieldSpec",
    "Skill",
    "RegistryClient",
    "RegistryUnavailable",
    "get_registry_client",
]
