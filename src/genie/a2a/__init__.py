"""Formal A2A (Agent2Agent) protocol support — built on the official ``a2a-sdk``
JSON models — layered on top of this framework's centralized Registry discovery.

Wire types come from ``a2a.compat.v0_3.types`` (the JSON binding of protocol 1.0.0);
see :mod:`genie.a2a.types`.
"""
from genie.a2a.agent_card import a2a_url, to_agent_card
from genie.a2a.client import A2AClient, A2AError
from genie.a2a.types import (
    METHOD_MESSAGE_SEND,
    METHOD_MESSAGE_STREAM,
    METHOD_TASKS_CANCEL,
    METHOD_TASKS_GET,
    PROTOCOL_VERSION,
    AgentCard,
    Artifact,
    DataPart,
    JSONRPCErrorResponse,
    Message,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    data_part,
    get_data,
    get_text,
    task_final_message,
    text_part,
)

__all__ = [
    "A2AClient",
    "A2AError",
    "AgentCard",
    "Artifact",
    "DataPart",
    "JSONRPCErrorResponse",
    "METHOD_MESSAGE_SEND",
    "METHOD_MESSAGE_STREAM",
    "METHOD_TASKS_CANCEL",
    "METHOD_TASKS_GET",
    "PROTOCOL_VERSION",
    "Message",
    "Role",
    "Task",
    "TaskArtifactUpdateEvent",
    "TaskState",
    "TaskStatus",
    "TaskStatusUpdateEvent",
    "TextPart",
    "a2a_url",
    "data_part",
    "get_data",
    "get_text",
    "task_final_message",
    "text_part",
    "to_agent_card",
]
