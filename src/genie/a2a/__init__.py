"""Formal A2A (Agent2Agent) protocol support — JSON-RPC ``message/send`` and
Agent Cards — layered on top of this framework's centralized Registry discovery.
"""
from genie.a2a.agent_card import a2a_url, to_agent_card
from genie.a2a.client import A2AClient, A2AError
from genie.a2a.types import (
    METHOD_MESSAGE_SEND,
    AgentCard,
    DataPart,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    Message,
    TextPart,
    data_part,
    get_data,
    get_text,
    text_part,
)

__all__ = [
    "A2AClient",
    "A2AError",
    "AgentCard",
    "DataPart",
    "JsonRpcError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "METHOD_MESSAGE_SEND",
    "Message",
    "TextPart",
    "a2a_url",
    "data_part",
    "get_data",
    "get_text",
    "text_part",
    "to_agent_card",
]
