"""Derive a formal A2A Agent Card from this framework's ``AgentMeta``.

The Registry remains the single source of truth for discovery; the Agent Card is
just the A2A-shaped *view* of an ``AgentMeta``, so there is no schema drift
between what an agent advertises in the registry and the card it serves at
``/.well-known/agent.json``.
"""
from __future__ import annotations

from genie.a2a.types import (
    PROTOCOL_VERSION,
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
)
from genie.platform.config import get_settings
from genie.registry.agent_meta import AgentMeta


def to_agent_card(meta: AgentMeta) -> AgentCard:
    """Map an :class:`AgentMeta` onto an A2A v1.2 :class:`AgentCard`.

    The card's ``skills`` are projected directly from ``meta.skills`` — the same
    list the registry stores (auto-derived from capability_tags + input_schema
    when an agent doesn't set them explicitly), so the card and the registry
    record always advertise identical skills.

    ``capabilities.streaming`` reflects ``meta.supports_streaming`` (default
    True — the harness serves ``message/stream`` for every agent, and gates that
    endpoint on the same flag so the card never over-promises). The single
    JSONRPC interface is echoed in ``additionalInterfaces`` so a gRPC/HTTP+JSON
    interface can be added later without a breaking change. Bearer auth is
    advertised only when the agent actually enforces a token (see
    :func:`_security`) — token-free agents serve a card with no security
    section, unchanged from before.
    """
    settings = get_settings()
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
    url = a2a_url(meta.endpoint)
    schemes, security = _security(settings.agent_invoke_token)
    return AgentCard(
        name=meta.agent_id,
        description=meta.description or "",
        url=url,
        version=meta.version,
        protocolVersion=settings.a2a_protocol_version or PROTOCOL_VERSION,
        additionalInterfaces=[AgentInterface(url=url)] if url else [],
        provider=_provider(settings),
        documentationUrl=meta.changelog_url,
        capabilities=AgentCapabilities(streaming=meta.supports_streaming, pushNotifications=False),
        securitySchemes=schemes,
        security=security,
        skills=skills,
    )


def _provider(settings) -> AgentProvider | None:
    """Build an :class:`AgentProvider` from config, or None when unconfigured."""
    org = settings.agent_provider_organization
    if not org:
        return None
    return AgentProvider(organization=org, url=settings.agent_provider_url or "")


def _security(
    token: str | None,
) -> tuple[dict[str, dict] | None, list[dict[str, list[str]]] | None]:
    """Advertise the HTTP bearer scheme only when the agent enforces a token.

    Returns ``(None, None)`` for token-free agents so their card carries no
    security section — the pre-1.2 behavior — and a spec-shaped bearer scheme +
    requirement otherwise, matching what ``/a2a`` actually enforces.
    """
    if not token:
        return None, None
    return {"bearer": {"type": "http", "scheme": "bearer"}}, [{"bearer": []}]


def a2a_url(endpoint: str | None) -> str:
    """The A2A JSON-RPC URL for an agent base endpoint (``/a2a`` by convention)."""
    base = (endpoint or "").rstrip("/")
    return f"{base}/a2a" if base else ""
