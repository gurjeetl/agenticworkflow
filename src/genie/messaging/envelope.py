"""Bus envelope helpers: topic names, headers, and deterministic correlation ids.

The Kafka message **value** is the same a2a-sdk ``Message``/``Task`` JSON used on
the synchronous wire; everything a router needs travels in **headers** so no
component has to deserialize a body to route, dedup, or dead-letter it.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from genie.platform.config import get_settings

# Fixed namespace for deterministic correlation ids (uuid5). The Executor re-runs
# its dispatch on interrupt-resume, so the id MUST be reproducible: the same
# (run_id, task_id, attempt) always yields the same correlation_id, making the
# re-produce a duplicate the consumer's dedup drops — not a second execution.
_CID_NAMESPACE = uuid.UUID("b3a486a0-51c7-4d8e-9c96-a2a0c0ffee11")

# Header names (values are UTF-8 strings on the wire).
HDR_CORRELATION_ID = "correlation_id"
HDR_ATTEMPT = "attempt"
HDR_KIND = "kind"                    # request | reply | step.cancelled | dead_letter
HDR_FROM = "from"
HDR_TO = "to"
HDR_REPLY_TO = "reply_to"
HDR_THREAD_ID = "thread_id"
HDR_RUN_ID = "run_id"
HDR_TASK_ID = "task_id"              # the plan subtask id this request belongs to
HDR_DEADLINE = "deadline"            # RFC3339 UTC
HDR_ERROR = "error"                  # dead-letter diagnostic
HDR_GROUP_ID = "group_id"            # fan-out group (one wave's bus dispatches resume together)
HDR_TRACE_ID = "trace_id"            # cross-hop trace correlation (currently = run_id)
HDR_TENANT_ID = "tenant_id"          # multi-tenancy scope (empty for single-tenant)

KIND_REQUEST = "request"
KIND_REPLY = "reply"
KIND_STEP_CANCELLED = "step.cancelled"
KIND_DEAD_LETTER = "dead_letter"


def correlation_id(run_id: str, task_id: str, attempt: int) -> str:
    """Deterministic correlation id for one dispatch attempt (see module note)."""
    return uuid.uuid5(_CID_NAMESPACE, f"{run_id}:{task_id}:{attempt}").hex


def inbox_topic(agent_id: str) -> str:
    """Default inbox topic for an agent (overridable via ``AgentMeta.inbox_topic``)."""
    return f"{get_settings().bus_topic_prefix}.agents.{agent_id}.inbox"


def reply_topic() -> str:
    """The shared reply/control topic the gateway's reply-router consumes."""
    s = get_settings()
    return s.bus_reply_topic or f"{s.bus_topic_prefix}.replies"


def dlq_topic() -> str:
    """The dead-letter topic (poison pills + retries exhausted; replayable)."""
    s = get_settings()
    return s.bus_dlq_topic or f"{s.bus_topic_prefix}.dlq"


def deadline_from_now(deadline_ms: int) -> str:
    """RFC3339 UTC deadline ``deadline_ms`` from now, for the request header."""
    return (datetime.now(timezone.utc) + timedelta(milliseconds=deadline_ms)).isoformat()


def parse_deadline(value: str | None) -> datetime | None:
    """Parse an RFC3339 header deadline back to an aware datetime (None-safe)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
