"""Canonical event-name constants for structured logging and span events.

Centralizing the dotted event names here keeps them consistent across agents,
tools, and the MCP layer so traces and log queries can filter on stable keys.
"""


class Events:
    """Namespace of dotted event-name string constants used in logs/spans."""

    FINAL_OUTPUT_SET = "final.output.set"
    AGENT_ERROR_SET = "agent.error_set"
    AGENT_SCRATCHPAD = "agent.scratchpad"

    FORMAT_MESSAGES = "format.messages"

    LLM_TOOL_CALLS = "llm.tool_calls"
    LLM_TOOL_RESULTS = "llm.tool_results"
    LLM_ERROR = "llm.error"
    LLM_INVOKE_FAILED = "llm.invoke_failed"

    TOOL_INVOKE_FAILED = "tool.invoke_failed"
    AGENT_RUN_FAILED = "agent.run_failed"

    MCP_TOOL_CALL = "mcp.tool_call"
    MCP_TOOL_FAILED = "mcp.tool_failed"
    MCP_UNKNOWN_TRANSPORT = "mcp.unknown_transport"
    MCP_TOOLS_LOADED = "mcp.tools_loaded"
    MCP_LOAD_FAILED = "mcp.load_failed"
