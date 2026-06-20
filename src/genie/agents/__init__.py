"""Agent SDK surface for applications.

Applications build concrete agents by inheriting :class:`BaseAgent`; the platform
provides LLM, MCP-tool connectivity, and working memory through it.
"""
from genie.agents.base import BaseAgent

__all__ = ["BaseAgent"]
