"""Unit tests for the AgentMeta → A2A v1.2 AgentCard projection."""
from __future__ import annotations

import pytest

import genie.platform.config as cfg
from genie.a2a.agent_card import to_agent_card
from genie.registry.agent_meta import AgentMeta, FieldSpec


META = AgentMeta(
    agent_id="stub",
    version="1.0.0",
    capability_tags=["test"],
    description="A stub agent.",
    input_schema={"location": FieldSpec(type="string", required=True)},
    endpoint="http://127.0.0.1:8010",
)


@pytest.fixture
def restore_settings():
    """Snapshot the settings singleton and restore it after the test."""
    base = cfg.get_settings()
    yield base
    cfg.override_settings(base)


def test_card_advertises_1_2_streaming_and_interface(restore_settings):
    base = restore_settings
    cfg.override_settings(base.model_copy(update={"agent_invoke_token": None}))
    card = to_agent_card(META)
    assert card.protocolVersion == "1.2"
    assert card.capabilities.streaming is True
    assert card.capabilities.pushNotifications is False
    assert [i.transport for i in card.additionalInterfaces] == ["JSONRPC"]
    assert card.additionalInterfaces[0].url.endswith("/a2a")


def test_card_omits_security_without_token(restore_settings):
    base = restore_settings
    cfg.override_settings(base.model_copy(update={"agent_invoke_token": None}))
    card = to_agent_card(META)
    assert card.securitySchemes is None
    assert card.security is None


def test_card_declares_bearer_when_token_set(restore_settings):
    base = restore_settings
    cfg.override_settings(base.model_copy(update={"agent_invoke_token": "secret"}))
    card = to_agent_card(META)
    assert card.securitySchemes == {"bearer": {"type": "http", "scheme": "bearer"}}
    assert card.security == [{"bearer": []}]


def test_card_provider_from_config(restore_settings):
    base = restore_settings
    cfg.override_settings(
        base.model_copy(update={"agent_provider_organization": "OATI", "agent_provider_url": "https://oati.example"})
    )
    card = to_agent_card(META)
    assert card.provider is not None
    assert card.provider.organization == "OATI"
