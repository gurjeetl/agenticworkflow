"""Tests for MCPClient.normalize_result — the spec-aligned tool-result projection."""
from langchain_core.messages import ToolMessage

from genie.mcp.client import MCPClient, MCPToolResult


def _msg(content, artifact=None, status="success") -> ToolMessage:
    """Build a ToolMessage like the one langchain returns from an MCP ToolCall."""
    return ToolMessage(
        content=content, artifact=artifact, status=status, tool_call_id="t1"
    )


def test_structured_dict_reaches_caller_parsed() -> None:
    # A dict tool: FastMCP sends a JSON text block AND structuredContent (artifact).
    msg = _msg(
        [{"type": "text", "text": '{"total": 2, "items": [1, 2]}'}],
        artifact={"structured_content": {"total": 2, "items": [1, 2]}},
    )
    r = MCPClient.normalize_result(msg)
    assert isinstance(r, MCPToolResult)
    assert r.structured == {"total": 2, "items": [1, 2]}
    assert r.is_error is False
    assert r.text  # the JSON text block is preserved for display/LLM


def test_result_wrapper_is_unwrapped() -> None:
    # FastMCP wraps non-object returns (list/primitive) as {"result": X}.
    msg = _msg("[1, 2, 3]", artifact={"structured_content": {"result": [1, 2, 3]}})
    assert MCPClient.normalize_result(msg).structured == [1, 2, 3]


def test_genuine_result_key_is_not_overunwrapped_when_multikey() -> None:
    # Only a *sole* {"result": ...} is unwrapped; a richer object is left intact.
    payload = {"result": 1, "other": 2}
    msg = _msg("x", artifact={"structured_content": payload})
    assert MCPClient.normalize_result(msg).structured == payload


def test_plain_text_tool() -> None:
    r = MCPClient.normalize_result(_msg("Sunny, 22C"))
    assert r.text == "Sunny, 22C"
    assert r.structured is None
    assert r.blocks == ()


def test_image_block_is_reference_without_base64() -> None:
    msg = _msg([{"type": "image", "base64": "QUJD", "mime_type": "image/png"}])
    r = MCPClient.normalize_result(msg)
    assert len(r.blocks) == 1
    blk = r.blocks[0]
    assert blk["type"] == "image"
    assert blk["mime_type"] == "image/png"
    assert blk["has_inline_data"] is True
    assert "base64" not in blk and "QUJD" not in str(blk)
    # No text block, so text is a short placeholder (never the base64 payload).
    assert r.text == "[image: image/png]"
    assert "QUJD" not in r.text


def test_file_resource_link_keeps_uri() -> None:
    msg = _msg([{"type": "file", "url": "https://x/y.pdf", "mime_type": "application/pdf"}])
    blk = MCPClient.normalize_result(msg).blocks[0]
    assert blk["uri"] == "https://x/y.pdf"
    assert blk["has_inline_data"] is False


def test_is_error_from_status() -> None:
    r = MCPClient.normalize_result(_msg("boom", status="error"))
    assert r.is_error is True


def test_legacy_shapes_still_normalize() -> None:
    assert MCPClient.normalize_result("hi").text == "hi"
    assert MCPClient.normalize_result({"a": 1}).structured == {"a": 1}
    assert MCPClient.normalize_result(
        [{"type": "text", "text": "x"}]
    ).text == "x"
    # Back-compat shim returns just the text.
    assert MCPClient.unwrap_result(_msg("plain")) == "plain"
