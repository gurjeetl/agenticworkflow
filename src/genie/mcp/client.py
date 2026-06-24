"""MCPClient: builds MCP server config from env/settings, loads tools, normalizes results.

This is the platform's MCP connectivity surfaced to inheriting agents — it turns
configured servers into permission-filtered LangChain tools.
"""
import os
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from genie.platform.config import get_settings
from genie.platform.events import Events
from genie.mcp.permissions import filter_tools_by_permission
from genie.mcp.config import MCPAgentConfig, MCPServerConfig, MCPTransport


@dataclass(frozen=True)
class MCPToolResult:
    """Normalized, spec-aligned view of an MCP ``CallToolResult``.

    - ``text``: the human/LLM-readable text projection (joined text blocks; a
      short placeholder when the result is binary-only). Never contains base64.
    - ``structured``: the parsed ``structuredContent`` object (dict/list), with
      FastMCP's ``{"result": X}`` wrapper for non-object returns unwrapped.
    - ``blocks``: non-text content (image/audio/file/resource) as lightweight
      by-reference descriptors ``{type, mime_type, uri, has_inline_data}`` —
      deliberately *without* the inline base64 payload.
    - ``is_error``: mirrors MCP ``isError`` (the tool reported an execution error).
    """
    text: str
    structured: Any = None
    blocks: tuple[dict, ...] = ()
    is_error: bool = False


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
    def normalize_result(result) -> MCPToolResult:
        """Project a tool result into a spec-aligned :class:`MCPToolResult`.

        The intended input is the langchain ``ToolMessage`` produced by invoking
        an MCP tool with a ToolCall payload — the only invocation shape that
        preserves ``structuredContent`` (it rides in the message ``artifact``).
        The legacy ``str``/``dict``/``list`` shapes are still accepted so any
        non-ToolMessage caller degrades gracefully to a text-only result.
        """
        # Legacy / raw shapes — keep working for robustness.
        if isinstance(result, str):
            return MCPToolResult(text=result)
        if isinstance(result, dict):
            return MCPToolResult(text="", structured=MCPClient._unwrap_structured(result))
        if isinstance(result, list):
            text, blocks = MCPClient._split_blocks(result)
            return MCPToolResult(text=text, blocks=tuple(blocks))

        # The normal path: a langchain ToolMessage.
        content = getattr(result, "content", None)
        artifact = getattr(result, "artifact", None)
        is_error = getattr(result, "status", None) == "error"

        structured = None
        if isinstance(artifact, dict):
            structured = MCPClient._unwrap_structured(artifact.get("structured_content"))

        if isinstance(content, str):
            text, blocks = content, []
        elif isinstance(content, list):
            text, blocks = MCPClient._split_blocks(content)
        else:
            text, blocks = "", []

        if not text and blocks:
            ref = blocks[0]
            label = ref.get("mime_type") or ref.get("uri") or "binary"
            text = f"[{ref.get('type') or 'content'}: {label}]"

        return MCPToolResult(
            text=text,
            structured=structured,
            blocks=tuple(blocks),
            is_error=is_error,
        )

    @staticmethod
    def unwrap_result(result) -> str:
        """Back-compat shim: return just the text projection of a tool result."""
        return MCPClient.normalize_result(result).text

    @staticmethod
    def _unwrap_structured(structured):
        """Unwrap FastMCP's ``{"result": X}`` envelope used for non-object returns."""
        if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured

    @staticmethod
    def _split_blocks(content: list) -> tuple[str, list[dict]]:
        """Split a content-block list into joined text and base64-free non-text refs."""
        text_parts: list[str] = []
        blocks: list[dict] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, dict):
                blocks.append(MCPClient._block_reference(item))
            else:
                text_parts.append(str(item))
        return " ".join(p for p in text_parts if p), blocks

    @staticmethod
    def _block_reference(block: dict) -> dict:
        """Reduce a non-text content block to a lightweight reference (no base64)."""
        return {
            "type": block.get("type"),
            "mime_type": block.get("mime_type"),
            "uri": block.get("url"),
            "has_inline_data": bool(block.get("base64")),
        }
