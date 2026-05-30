import os
from typing import Protocol

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from baseagent.events import Events
from baseagent.permissions import filter_tools_by_permission
from mcpconfig.mcp_config import MCPAgentConfig, MCPServerConfig, MCPTransport


class _Observer(Protocol):
    def log(self, level: str, event: str, **attrs) -> None: ...


class MCPClient:
    """Builds MCP configuration from env, loads tools, and unwraps results."""

    def __init__(self, observer: _Observer) -> None:
        self._observer = observer

    def build_config_from_env(self) -> MCPAgentConfig | None:
        """Construct an MCPAgentConfig from environment variables.

        Reads MCP_SERVER_URL (required), MCP_SERVER_NAME, MCP_TRANSPORT,
        MCP_AUTH_TOKEN, MCP_TIMEOUT.
        """
        url = os.getenv("MCP_SERVER_URL")
        if not url:
            return None

        transport_str = os.getenv("MCP_TRANSPORT", "sse")
        try:
            transport = MCPTransport(transport_str)
        except ValueError:
            self._observer.log(
                "warning",
                Events.MCP_UNKNOWN_TRANSPORT,
                transport=transport_str,
                fallback="sse",
            )
            transport = MCPTransport.SSE

        token = os.getenv("MCP_AUTH_TOKEN", "")
        server = MCPServerConfig(
            name=os.getenv("MCP_SERVER_NAME", "default"),
            url=url,
            transport=transport,
            timeout=float(os.getenv("MCP_TIMEOUT", "30.0")),
            headers={"Authorization": f"Bearer {token}"} if token else {},
        )
        return MCPAgentConfig(servers=[server])

    async def load_tools(
        self,
        config: MCPAgentConfig,
        tool_names: list[str] | None,
    ) -> list[BaseTool]:
        server_map = {
            s.name: {
                "transport": s.transport.value,
                "url": s.url,
                "headers": s.headers,
                "timeout": s.timeout,
            }
            for s in config.servers
        }
        client = MultiServerMCPClient(server_map)
        tools = await client.get_tools()
        if tool_names is not None:
            allowed = set(tool_names)
            tools = [t for t in tools if t.name in allowed]
        return filter_tools_by_permission(tools)

    @staticmethod
    def unwrap_result(result) -> str:
        """MCP tool results may come back as a list of content blocks; flatten to text."""
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            parts: list[str] = []
            for item in result:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                else:
                    parts.append(str(item))
            return " ".join(parts)
        return str(result)
