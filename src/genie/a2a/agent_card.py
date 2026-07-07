"""Derive a formal A2A Agent Card from this framework's ``AgentMeta``.

The Registry remains the single source of truth for discovery; the Agent Card is
just the A2A-shaped *view* of an ``AgentMeta``, built with the official ``a2a-sdk``
models so there is no schema drift between what an agent advertises in the registry
and the card it serves at ``/.well-known/agent-card.json``.
"""
from __future__ import annotations

from genie.a2a.types import (
    PROTOCOL_VERSION,
    TRANSPORT_JSONRPC,
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    HTTPAuthSecurityScheme,
    SecurityScheme,
)
from genie.platform.config import get_settings
from genie.registry.agent_meta import AgentMeta


def to_agent_card(meta: AgentMeta) -> AgentCard:
    """Map an :class:`AgentMeta` onto an A2A :class:`AgentCard` (a2a-sdk model).

    ``skills`` are projected directly from ``meta.skills`` — the same list the
    registry stores — so the card and the registry record always advertise
    identical skills. ``capabilities.streaming`` reflects ``meta.supports_streaming``
    (the harness serves ``message/stream`` for streaming-capable agents and gates
    the endpoint on the same flag). The single JSONRPC interface is echoed in
    ``additional_interfaces`` so a gRPC/HTTP+JSON interface can be added later
    without a breaking change. Bearer auth is advertised only when the agent
    enforces a token (see :func:`_security`) — token-free agents serve a card with
    no security section, unchanged from before.
    """
    settings = get_settings()
    skills = [
        AgentSkill(
            id=s.id,
            name=s.name,
            description=s.description or "",
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
        protocol_version=settings.a2a_protocol_version or PROTOCOL_VERSION,
        preferred_transport=TRANSPORT_JSONRPC,
        additional_interfaces=[AgentInterface(url=url, transport=TRANSPORT_JSONRPC)] if url else [],
        provider=_provider(settings),
        documentation_url=meta.changelog_url,
        capabilities=AgentCapabilities(streaming=meta.supports_streaming, push_notifications=False),
        security_schemes=schemes,
        security=security,
        default_input_modes=["text", "data"],
        default_output_modes=["text", "data"],
        skills=skills,
    )


def _provider(settings) -> AgentProvider | None:
    """Build an :class:`AgentProvider` from config, or None when unconfigured."""
    org = settings.agent_provider_organization
    if not org:
        return None
    return AgentProvider(organization=org, url=settings.agent_provider_url or "")


def _security(token: str | None):
    """Advertise the HTTP bearer scheme only when the agent enforces a token.

    Returns ``(None, None)`` for token-free agents so their card carries no
    security section — the pre-adoption behavior — and a spec-shaped bearer scheme
    + requirement otherwise, matching what ``/a2a`` actually enforces.
    """
    if not token:
        return None, None
    scheme = SecurityScheme(root=HTTPAuthSecurityScheme(type="http", scheme="bearer"))
    return {"bearer": scheme}, [{"bearer": []}]


def a2a_url(endpoint: str | None) -> str:
    """The A2A JSON-RPC URL for an agent base endpoint (``/a2a`` by convention)."""
    base = (endpoint or "").rstrip("/")
    return f"{base}/a2a" if base else ""
