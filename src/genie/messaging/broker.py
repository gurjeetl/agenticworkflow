"""Broker abstraction for the async A2A transport.

``Broker`` is the minimal produce/consume surface the platform needs;
``KafkaBroker`` implements it with aiokafka (works against Redpanda locally and
corporate Apache Kafka via the security settings — a pure config change);
``FakeBroker`` is the in-memory implementation the test suite runs on, so no
test needs Docker or a real broker.

Loop-binding note: aiokafka clients are bound to the event loop they start on.
Long-lived loops (the agent process, the gateway) reuse a cached producer; the
Executor runs coroutines on transient per-call loops (see
``Observable._run_async``), so it produces with ``oneshot=True`` — a
create/send/stop producer scoped to the call.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

from genie.observability import get_logger
from genie.platform.config import get_settings

_log = get_logger(__name__)


@dataclass
class BusMessage:
    """One consumed bus record: topic, partition key, raw value, string headers."""

    topic: str
    key: str | None
    value: bytes
    headers: dict[str, str] = field(default_factory=dict)


class Broker(Protocol):
    """Minimal produce/consume contract the platform codes against."""

    async def produce(
        self, topic: str, *, value: bytes, key: str | None = None,
        headers: dict[str, str] | None = None, oneshot: bool = False,
    ) -> None:
        """Publish one record. ``oneshot`` = use a producer scoped to this call."""
        ...

    def consume(self, topics: list[str], *, group: str) -> AsyncIterator[BusMessage]:
        """Yield records for ``topics`` as part of consumer ``group`` (runs forever)."""
        ...

    async def close(self) -> None:
        """Release clients. Idempotent."""
        ...


# ---------------------------------------------------------------------------
def _security_kwargs() -> dict:
    """aiokafka security kwargs from Settings — corporate Kafka is config-only."""
    s = get_settings()
    kwargs: dict = {
        "bootstrap_servers": s.kafka_bootstrap_servers,
        "security_protocol": s.kafka_security_protocol,
    }
    if s.kafka_sasl_mechanism:
        kwargs["sasl_mechanism"] = s.kafka_sasl_mechanism
        kwargs["sasl_plain_username"] = s.kafka_sasl_username
        kwargs["sasl_plain_password"] = s.kafka_sasl_password
    if s.kafka_ssl_cafile:
        import ssl

        kwargs["ssl_context"] = ssl.create_default_context(cafile=s.kafka_ssl_cafile)
    return kwargs


def _encode_headers(headers: dict[str, str] | None) -> list[tuple[str, bytes]]:
    """Kafka wants headers as (str, bytes) pairs."""
    return [(k, str(v).encode("utf-8")) for k, v in (headers or {}).items()]


def _decode_headers(raw) -> dict[str, str]:
    """Decode Kafka (str, bytes) header pairs back to a string dict."""
    out: dict[str, str] = {}
    for k, v in raw or []:
        out[k] = v.decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray)) else str(v)
    return out


class KafkaBroker:
    """aiokafka-backed broker (Redpanda locally, Apache Kafka in production)."""

    def __init__(self) -> None:
        """Producers are created lazily per event loop; consumers per consume() call."""
        self._producers: dict[int, "object"] = {}  # loop id → started AIOKafkaProducer
        self._consumers: list = []

    async def _get_producer(self):
        """Start (or reuse) the producer bound to the current running loop."""
        from aiokafka import AIOKafkaProducer

        loop_id = id(asyncio.get_running_loop())
        producer = self._producers.get(loop_id)
        if producer is None:
            producer = AIOKafkaProducer(**_security_kwargs())
            await producer.start()
            self._producers[loop_id] = producer
        return producer

    async def produce(
        self, topic: str, *, value: bytes, key: str | None = None,
        headers: dict[str, str] | None = None, oneshot: bool = False,
    ) -> None:
        """Publish one record; ``oneshot`` scopes a producer to this call (transient loops)."""
        from aiokafka import AIOKafkaProducer

        kkey = key.encode("utf-8") if key else None
        kheaders = _encode_headers(headers)
        if oneshot:
            producer = AIOKafkaProducer(**_security_kwargs())
            await producer.start()
            try:
                await producer.send_and_wait(topic, value=value, key=kkey, headers=kheaders)
            finally:
                await producer.stop()
            return
        producer = await self._get_producer()
        await producer.send_and_wait(topic, value=value, key=kkey, headers=kheaders)

    async def consume(self, topics: list[str], *, group: str) -> AsyncIterator[BusMessage]:
        """Consume ``topics`` in ``group`` forever (auto-commit; dedup makes redelivery safe)."""
        from aiokafka import AIOKafkaConsumer

        consumer = AIOKafkaConsumer(
            *topics,
            group_id=group,
            enable_auto_commit=True,
            auto_offset_reset="earliest",
            **_security_kwargs(),
        )
        await consumer.start()
        self._consumers.append(consumer)
        try:
            async for rec in consumer:
                yield BusMessage(
                    topic=rec.topic,
                    key=rec.key.decode("utf-8") if rec.key else None,
                    value=rec.value,
                    headers=_decode_headers(rec.headers),
                )
        finally:
            await consumer.stop()

    async def close(self) -> None:
        """Stop every producer/consumer this broker started. Idempotent."""
        for producer in self._producers.values():
            try:
                await producer.stop()
            except Exception:
                pass
        self._producers.clear()
        for consumer in self._consumers:
            try:
                await consumer.stop()
            except Exception:
                pass
        self._consumers.clear()


class FakeBroker:
    """In-memory Broker for tests: per-(topic, group) queues + a produce log.

    ``log[topic]`` records every produced message for assertions. ``consume``
    yields from an asyncio.Queue, so tests drive consumers by producing first
    (or concurrently) and cancelling the consumer task when done.
    """

    def __init__(self) -> None:
        """Empty log and no subscribers."""
        self.log: dict[str, list[BusMessage]] = {}
        self._queues: dict[tuple[str, str], asyncio.Queue] = {}

    async def produce(
        self, topic: str, *, value: bytes, key: str | None = None,
        headers: dict[str, str] | None = None, oneshot: bool = False,
    ) -> None:
        """Record the message and fan it out to every group subscribed to ``topic``."""
        msg = BusMessage(topic=topic, key=key, value=value, headers=dict(headers or {}))
        self.log.setdefault(topic, []).append(msg)
        for (t, _g), q in self._queues.items():
            if t == topic:
                q.put_nowait(msg)

    async def consume(self, topics: list[str], *, group: str) -> AsyncIterator[BusMessage]:
        """Yield messages for ``topics`` (backlog first, then live) until cancelled."""
        queue: asyncio.Queue = asyncio.Queue()
        for topic in topics:
            self._queues[(topic, group)] = queue
            for msg in self.log.get(topic, []):
                queue.put_nowait(msg)  # deliver backlog produced before subscribing
        while True:
            yield await queue.get()

    async def close(self) -> None:
        """Drop all subscriptions."""
        self._queues.clear()


# ---------------------------------------------------------------------------
_broker: Broker | None = None


def get_broker() -> Broker:
    """Process-wide broker singleton (KafkaBroker unless a test injected a fake)."""
    global _broker
    if _broker is None:
        _broker = KafkaBroker()
    return _broker


def set_broker(broker: Broker | None) -> None:
    """Inject a Broker (tests) or reset to lazy default with ``None``."""
    global _broker
    _broker = broker
