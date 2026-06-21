"""LLMClient: wraps the ChatOpenAI handle with message construction and tool execution.

Centralizes invoke/error-logging and the async tool-call fan-out so BaseAgent and
its subclasses don't repeat the LangChain plumbing.
"""
import asyncio
from typing import Protocol

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from genie.platform.events import Events


class _Observer(Protocol):
    """Structural type for the host (the agent) that receives logs/events."""
    def log(self, level: str, event: str, **attrs) -> None: ...
    def log_event(self, name: str, **attrs) -> None: ...


class LLMClient:
    """Owns the ChatOpenAI handle, message construction, and tool execution."""

    def __init__(self, llm: ChatOpenAI, observer: _Observer) -> None:
        self.llm = llm
        self.tools: list[BaseTool] = []
        self._observer = observer

    def bind_tools(self, tools: list[BaseTool]) -> None:
        """Register tools for execution and bind them to the model for tool-calling."""
        self.tools = tools
        if tools:
            self.llm = self.llm.bind_tools(tools)

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Call the model, logging and re-raising on failure (the caller decides recovery)."""
        try:
            return self.llm.invoke(messages)
        except Exception as e:
            self._observer.log(
                "error",
                Events.LLM_INVOKE_FAILED,
                agent=type(self._observer).__name__,
                error=str(e),
                exc_info=True,
            )
            self._observer.log_event(Events.LLM_ERROR, error=str(e))
            raise

    def call(self, messages: list[BaseMessage]) -> str:
        """Invoke the model and return just the text content."""
        return self.invoke(messages).content

    async def execute_tool_calls(self, tool_calls: list[dict]) -> list[ToolMessage]:
        """Run the model's requested tool calls concurrently, one ToolMessage each.

        Unknown tools and tool exceptions are turned into error ToolMessages rather
        than raising, so a single bad call doesn't abort the whole batch.
        """
        tool_map = {t.name: t for t in self.tools}

        async def _call_one(tc: dict) -> ToolMessage:
            tool = tool_map.get(tc["name"])
            if tool is None:
                return ToolMessage(
                    content=f"Tool '{tc['name']}' not found.",
                    tool_call_id=tc["id"],
                )
            try:
                result = await tool.ainvoke(tc["args"])
                return ToolMessage(content=str(result), tool_call_id=tc["id"])
            except Exception as e:
                self._observer.log(
                    "error",
                    Events.TOOL_INVOKE_FAILED,
                    tool=tc["name"],
                    error=str(e),
                    exc_info=True,
                )
                return ToolMessage(
                    content=f"Error calling '{tc['name']}': {e}",
                    tool_call_id=tc["id"],
                )

        return list(await asyncio.gather(*(_call_one(tc) for tc in tool_calls)))

    @staticmethod
    def build_messages(
        system_prompt: str,
        trimmed: list[BaseMessage],
        facts_block: str,
    ) -> list[BaseMessage]:
        """Assemble the LangChain message list: system prompt (+ facts), then history.

        Plain-dict history entries are coerced to Human/AI messages by their role.
        """
        prompt = system_prompt
        if facts_block:
            prompt = f"{prompt}\n\n## Known context about this user:\n{facts_block}"

        lc_messages: list[BaseMessage] = []
        if prompt:
            lc_messages.append(SystemMessage(content=prompt))
        for msg in trimmed:
            if isinstance(msg, (HumanMessage, AIMessage)):
                lc_messages.append(msg)
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                lc_messages.append(
                    HumanMessage(content=content) if role == "user" else AIMessage(content=content)
                )
        return lc_messages
