"""A2A wire types — thin adapter over the official ``a2a-sdk`` pydantic models.

The A2A JSON/REST wire types are the official ``a2a-sdk`` models from
``a2a.compat.v0_3.types`` — the JSON binding of A2A protocol **1.0.0**, advertised
as ``protocolVersion 0.3.0`` (the SDK-native value its client and the A2A Inspector
expect for JSON). Building on the SDK guarantees structural conformance and the
**standard JSON-RPC error codes** (``TaskNotFoundError = -32001``,
``TaskNotCancelableError = -32002``, ``UnsupportedOperationError = -32004``, …) and
removes the drift risk of hand-maintaining our own subset.

This module re-exports the SDK models under the names the rest of ``genie`` already
imports, plus a few genie-specific helpers. There is intentionally **no** hand-rolled
JSON-RPC envelope or error-code constant here anymore — the SDK owns those.
"""
from __future__ import annotations

from typing import Any

from a2a.compat.v0_3.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Artifact,
    ContentTypeNotSupportedError,
    DataPart,
    HTTPAuthSecurityScheme,
    InternalError,
    InvalidAgentResponseError,
    InvalidParamsError,
    JSONRPCError,
    JSONRPCErrorResponse,
    Message,
    MethodNotFoundError,
    Part,
    PushNotificationNotSupportedError,
    Role,
    SecurityScheme,
    Task,
    TaskArtifactUpdateEvent,
    TaskNotCancelableError,
    TaskNotFoundError,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)

# --- Version / transport / method constants ---------------------------------
# SDK-native protocolVersion for the JSON binding (== JSON encoding of protocol
# 1.0.0). Overridable at the card level via Settings.a2a_protocol_version.
PROTOCOL_VERSION = "0.3.0"
TRANSPORT_JSONRPC = "JSONRPC"

METHOD_MESSAGE_SEND = "message/send"
METHOD_MESSAGE_STREAM = "message/stream"
METHOD_TASKS_GET = "tasks/get"
METHOD_TASKS_CANCEL = "tasks/cancel"


# --- Helpers ----------------------------------------------------------------
def text_part(text: str) -> Part:
    """Wrap free text in an A2A :class:`Part` (the SDK's RootModel union)."""
    return Part(root=TextPart(text=text))


def data_part(data: dict[str, Any]) -> Part:
    """Wrap a structured object in an A2A :class:`Part`."""
    return Part(root=DataPart(data=data))


def get_text(message: Message) -> str:
    """Concatenate the text of every text part in the message."""
    chunks = [p.root.text for p in message.parts if isinstance(p.root, TextPart)]
    return "\n".join(c for c in chunks if c)


def get_data(message: Message) -> dict[str, Any]:
    """Merge the ``data`` of every data part in the message (later parts win)."""
    merged: dict[str, Any] = {}
    for p in message.parts:
        if isinstance(p.root, DataPart):
            merged.update(p.root.data)
    return merged


def task_final_message(task: Task) -> Message:
    """Extract the agent's reply Message from a completed :class:`Task`.

    Prefers ``status.message``; falls back to assembling a Message from the task's
    artifact parts so a caller (e.g. the internal :class:`A2AClient`) always gets a
    :class:`Message`. Raises ``ValueError`` if the task carries no content.
    """
    if task.status.message is not None:
        return task.status.message
    parts: list[Part] = []
    for artifact in task.artifacts or []:
        parts.extend(artifact.parts)
    if not parts:
        raise ValueError(f"task '{task.id}' has no message or artifact content")
    return Message(role=Role.agent, message_id=task.id, parts=parts)


__all__ = [
    # domain models
    "AgentCapabilities",
    "AgentCard",
    "AgentInterface",
    "AgentProvider",
    "AgentSkill",
    "Artifact",
    "DataPart",
    "HTTPAuthSecurityScheme",
    "Message",
    "Part",
    "Role",
    "SecurityScheme",
    "Task",
    "TaskArtifactUpdateEvent",
    "TaskState",
    "TaskStatus",
    "TaskStatusUpdateEvent",
    "TextPart",
    # errors / envelopes
    "ContentTypeNotSupportedError",
    "InternalError",
    "InvalidAgentResponseError",
    "InvalidParamsError",
    "JSONRPCError",
    "JSONRPCErrorResponse",
    "MethodNotFoundError",
    "PushNotificationNotSupportedError",
    "TaskNotCancelableError",
    "TaskNotFoundError",
    "UnsupportedOperationError",
    # constants
    "METHOD_MESSAGE_SEND",
    "METHOD_MESSAGE_STREAM",
    "METHOD_TASKS_CANCEL",
    "METHOD_TASKS_GET",
    "PROTOCOL_VERSION",
    "TRANSPORT_JSONRPC",
    # helpers
    "data_part",
    "get_data",
    "get_text",
    "task_final_message",
    "text_part",
]
