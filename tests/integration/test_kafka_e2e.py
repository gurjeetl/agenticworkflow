"""End-to-end broker test against a REAL Kafka/Redpanda (gated, skipped by default).

Enable with a running broker (e.g. ``docker compose up -d redpanda``) and::

    KAFKA_BOOTSTRAP_SERVERS=localhost:9092 uv run pytest -m kafka

Proves KafkaBroker's produce/consume/headers round-trip on the actual wire —
the rest of the async path is covered by the FakeBroker suites, which exercise
identical code above the Broker interface.
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest

import genie.platform.config as cfg
from genie.messaging.broker import KafkaBroker

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS")

pytestmark = [
    pytest.mark.kafka,
    pytest.mark.skipif(not BOOTSTRAP, reason="KAFKA_BOOTSTRAP_SERVERS not set — no real broker available"),
]


async def test_kafka_broker_round_trip_with_headers():
    base = cfg.get_settings()
    cfg.override_settings(base.model_copy(update={"kafka_bootstrap_servers": BOOTSTRAP}))
    broker = KafkaBroker()
    topic = f"genie.test.{uuid.uuid4().hex}"
    try:
        await broker.produce(
            topic,
            value=b'{"hello": "kafka"}',
            key="thr-1",
            headers={"correlation_id": "cid-e2e", "kind": "request"},
        )
        gen = broker.consume([topic], group=f"e2e-{uuid.uuid4().hex}")
        msg = await asyncio.wait_for(gen.__anext__(), timeout=20)
        assert msg.value == b'{"hello": "kafka"}'
        assert msg.key == "thr-1"
        assert msg.headers["correlation_id"] == "cid-e2e"
        assert msg.headers["kind"] == "request"
    finally:
        await broker.close()
        cfg.override_settings(base)
