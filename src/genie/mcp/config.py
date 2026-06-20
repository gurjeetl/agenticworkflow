from dataclasses import dataclass, field
from enum import Enum


class MCPTransport(str, Enum):
    SSE = "sse"
    STDIO = "stdio"
    WEBSOCKET = "websocket"
    STREAMABLE_HTTP = "streamable_http"


@dataclass
class MCPServerConfig:
    name: str
    url: str
    transport: MCPTransport = MCPTransport.SSE
    timeout: float = 30.0
    retries: int = 3
    headers: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)


@dataclass
class MCPAgentConfig:
    servers: list[MCPServerConfig]
    default_timeout: int = 30
    allowed_roles: list[str] = field(default_factory=list)
