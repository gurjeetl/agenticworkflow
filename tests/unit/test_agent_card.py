"""Unit tests for the AgentMeta → a2a-sdk AgentCard projection."""
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


def test_card_advertises_sdk_native_version_and_interface(restore_settings):
    base = restore_settings
    cfg.override_settings(base.model_copy(update={"agent_invoke_token": None}))
    card = to_agent_card(META)
    assert card.protocol_version == "0.3.0"
    assert card.capabilities.streaming is True
    assert card.capabilities.push_notifications is False
    assert [i.transport for i in card.additional_interfaces] == ["JSONRPC"]
    assert card.additional_interfaces[0].url.endswith("/a2a")


def test_card_omits_security_without_token(restore_settings):
    base = restore_settings
    cfg.override_settings(base.model_copy(update={"agent_invoke_token": None}))
    card = to_agent_card(META)
    assert card.security_schemes is None
    assert card.security is None


def test_card_declares_bearer_when_token_set(restore_settings):
    base = restore_settings
    cfg.override_settings(base.model_copy(update={"agent_invoke_token": "secret"}))
    card = to_agent_card(META)
    dumped = card.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert dumped["securitySchemes"] == {"bearer": {"type": "http", "scheme": "bearer"}}
    assert dumped["security"] == [{"bearer": []}]


def test_card_streaming_follows_meta_flag(restore_settings):
    base = restore_settings
    cfg.override_settings(base.model_copy(update={"agent_invoke_token": None}))
    no_stream = META.model_copy(update={"supports_streaming": False})
    assert to_agent_card(no_stream).capabilities.streaming is False
    assert to_agent_card(META).capabilities.streaming is True


def test_card_provider_from_config(restore_settings):
    base = restore_settings
    cfg.override_settings(
        base.model_copy(update={"agent_provider_organization": "OATI", "agent_provider_url": "https://oati.example"})
    )
    card = to_agent_card(META)
    assert card.provider is not None
    assert card.provider.organization == "OATI"
