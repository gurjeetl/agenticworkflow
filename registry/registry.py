from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from registry.agent_meta import AgentMeta

if TYPE_CHECKING:
    from baseagent.base_agent import BaseAgent

_log = logging.getLogger(__name__)

AGENT_REGISTRY: dict[str, tuple[AgentMeta, type]] = {}


def register(meta: AgentMeta, cls: type) -> None:
    if meta.agent_id in AGENT_REGISTRY:
        existing_meta, existing_cls = AGENT_REGISTRY[meta.agent_id]
        if existing_cls is cls and existing_meta.version == meta.version:
            return
        _log.warning(
            "registry.duplicate_agent_id",
            extra={"attrs": {"agent_id": meta.agent_id, "existing_version": existing_meta.version, "new_version": meta.version}},
        )
    AGENT_REGISTRY[meta.agent_id] = (meta, cls)


def get(agent_id: str) -> tuple[AgentMeta, type] | None:
    return AGENT_REGISTRY.get(agent_id)


def list_active() -> list[tuple[AgentMeta, type]]:
    return [(m, c) for m, c in AGENT_REGISTRY.values() if m.status == "active"]


def match_by_capability(tag: str) -> list[tuple[AgentMeta, type]]:
    return [(m, c) for m, c in AGENT_REGISTRY.values() if m.status == "active" and tag in m.capability_tags]
