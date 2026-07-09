"""Unit tests for the bus envelope: deterministic cids, topics, deadlines."""
from __future__ import annotations

from datetime import datetime, timezone

from genie.messaging.envelope import (
    correlation_id,
    deadline_from_now,
    dlq_topic,
    inbox_topic,
    parse_deadline,
    reply_topic,
)


def test_correlation_id_is_deterministic():
    """Re-executing the interrupted dispatch MUST reproduce the same cid."""
    a = correlation_id("run-1", "t1", 1)
    b = correlation_id("run-1", "t1", 1)
    assert a == b


def test_correlation_id_changes_per_attempt_and_task():
    base = correlation_id("run-1", "t1", 1)
    assert correlation_id("run-1", "t1", 2) != base  # retry = new attempt = new cid...
    assert correlation_id("run-1", "t2", 1) != base
    assert correlation_id("run-2", "t1", 1) != base


def test_topic_names_derive_from_prefix():
    assert inbox_topic("weather") == "genie.agents.weather.inbox"
    assert reply_topic() == "genie.replies"
    assert dlq_topic() == "genie.dlq"


def test_deadline_round_trip():
    header = deadline_from_now(60000)
    parsed = parse_deadline(header)
    assert parsed is not None
    assert parsed > datetime.now(timezone.utc)
    assert parse_deadline(None) is None
    assert parse_deadline("not-a-date") is None
