"""Tests that the sample MCP tools follow the spec: output schemas + isError."""
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from services.mcp import genie_mcp_server as server


async def test_all_tools_advertise_output_schema() -> None:
    tools = await server.mcp.list_tools()
    assert tools
    for t in tools:
        assert t.outputSchema is not None, f"{t.name} has no outputSchema"
        assert t.annotations and t.annotations.readOnlyHint is True


async def test_list_outage_ids_returns_structured_object() -> None:
    _content, structured = await server.mcp.call_tool("list_outage_ids", {})
    assert set(structured) >= {"total", "returned", "items"}
    assert isinstance(structured["items"], list)


async def test_weather_structured_and_unknown_city_errors() -> None:
    _content, structured = await server.mcp.call_tool("get_weather", {"city": "Paris"})
    assert structured == {"city": "paris", "report": server.WEATHER_DATA["paris"]}
    with pytest.raises(ToolError):
        await server.mcp.call_tool("get_weather", {"city": "atlantis"})


async def test_missing_outage_raises_tool_error() -> None:
    with pytest.raises(ToolError):
        await server.mcp.call_tool("get_outage_metadata", {"outage_id": -999})


async def test_linked_outages_is_wrapped_object() -> None:
    _content, structured = await server.mcp.call_tool("get_linked_outages", {})
    assert set(structured) == {"total", "items"}
    assert structured["total"] == len(structured["items"])
