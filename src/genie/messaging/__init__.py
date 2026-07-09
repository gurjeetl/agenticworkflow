"""Async A2A messaging: broker abstraction, envelope, dedup, awaiting records.

The bus carries the same a2a-sdk ``Message``/``Task`` JSON as the synchronous
wire; this package adds the transport around it — topics, headers, deterministic
correlation ids, idempotent-consumption claims, and the durable record of which
suspended run awaits which reply.
"""
from genie.messaging.awaiting import AwaitingStore, get_awaiting_store, set_awaiting_store
from genie.messaging.broker import Broker, BusMessage, FakeBroker, KafkaBroker, get_broker, set_broker
from genie.messaging.dedup import Dedup
from genie.messaging import envelope

__all__ = [
    "AwaitingStore",
    "Broker",
    "BusMessage",
    "Dedup",
    "FakeBroker",
    "KafkaBroker",
    "envelope",
    "get_awaiting_store",
    "get_broker",
    "set_awaiting_store",
    "set_broker",
]
