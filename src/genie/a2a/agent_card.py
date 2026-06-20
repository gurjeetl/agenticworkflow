"""Derive a formal A2A Agent Card from this framework's ``AgentMeta``.

The Registry remains the single source of truth for discovery; the Agent Card is
just the A2A-shaped *view* of an ``AgentMeta``, so there is no schema drift
between what an agent advertises in the registry and the card it serves at
``/.well-known/agent.json``.
"""
from __future__ import annotations

from genie.a2a.types import AgentCapabilities, AgentCard, AgentSkill
from genie.registry.agent_meta import AgentMeta


def to_agent_card(meta: AgentMeta) -> AgentCard:
    """Map an :class:`AgentMeta` onto an A2A :class:`AgentCard`.

    The card's ``skills`` are projected directly from ``meta.skills`` — the same
    list the registry stores (auto-derived from capability_tags + input_schema
    when an agent doesn't set them explicitly), so the card and the registry
    record always advertise identical skills.
    """
    skills = [
        AgentSkill(
            id=s.id,
            name=s.name,
            description=s.description,
            tags=list(s.tags),
            examples=s.examples,
        )
        for s in meta.skills
    ]
    return AgentCard(
        name=meta.agent_id,
        description=meta.description or "",
        url=a2a_url(meta.endpoint),
        version=meta.version,
        capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
        skills=skills,
    )


def a2a_url(endpoint: str | None) -> str:
    """The A2A JSON-RPC URL for an agent base endpoint (``/a2a`` by convention)."""
    base = (endpoint or "").rstrip("/")
    return f"{base}/a2a" if base else ""
