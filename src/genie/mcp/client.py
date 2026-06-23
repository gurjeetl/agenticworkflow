"""MCPClient: builds MCP server config from env/settings, loads tools, unwraps results.

This is the platform's MCP connectivity surfaced to inheriting agents — it turns
configured servers into permission-filtered LangChain tools.
"""
import os
from typing import Protocol

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from genie.platform.config import get_settings
from genie.platform.events import Events
from genie.mcp.permissions import filter_tools_by_permission
from genie.mcp.config import MCPAgentConfig, MCPServerConfig, MCPTransport


class _Observer(Protocol):
    """Structural type for the host (the agent) that receives warning/error logs."""
    def log(self, level: str, event: str, **attrs) -> None:
        """Receive a leveled log record with structured attributes."""
        ...


class MCPClient:
    """Builds MCP configuration from env, loads tools, and unwraps results."""

    def __init__(self, observer: _Observer) -> None:
        """Hold the observer used for warning/error logging."""
        self._observer = observer

    def build_config_from_env(self) -> MCPAgentConfig | None:
        """Construct an MCPAgentConfig from the central platform Settings.

        Sources two kinds of MCP server, both surfaced to inheriting agents:
        - the single flat server (MCP_SERVER_URL + MCP_SERVER_NAME / MCP_TRANSPORT /
          MCP_AUTH_TOKEN / MCP_TIMEOUT), and
        - any named servers declared under ``mcp_services`` in the YAML config.
        Returns None when neither is configured.
        """
        settings = get_settings()
        servers: list[MCPServerConfig] = []

        if settings.mcp_server_url:
            servers.append(self._build_server(
                name=settings.mcp_server_name,
                url=settings.mcp_server_url,
                transport_str=settings.mcp_transport,
                token=settings.mcp_auth_token,
                timeout=settings.mcp_timeout,
            ))

        for key, svc in settings.mcp_services.items():
            servers.append(self._build_server(
                name=svc.name if svc.name != "default" else key,
                url=svc.url,
                transport_str=svc.transport,
                token=svc.auth_token,
                timeout=svc.timeout,
            ))

        if not servers:
            return None
        return MCPAgentConfig(servers=servers)

    def _build_server(
        self, *, name: str, url: str, transport_str: str, token: str, timeout: float
    ) -> MCPServerConfig:
        """Build one MCPServerConfig, falling back to SSE for an unknown transport string."""
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
        return MCPServerConfig(
            name=name,
            url=url,
            transport=transport,
            timeout=timeout,
            headers={"Authorization": f"Bearer {token}"} if token else {},
        )

    async def load_tools(
        self,
        config: MCPAgentConfig,
        tool_names: list[str] | None,
    ) -> list[BaseTool]:
        """Connect to the configured servers and return the usable tool set.

        When ``tool_names`` is given, narrow to those names first; the result is
        then run through permission filtering before being handed to the agent.
        """
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
