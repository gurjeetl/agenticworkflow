import asyncio
import os
from typing import Callable

import mlflow
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from mlflow.entities import SpanType

from baseagent.agent_memory import AgentMemory
from baseagent.events import Events
from baseagent.llm_client import LLMClient
from baseagent.mcp_client import MCPClient
from observability import Observable
from state import AgentState

load_dotenv()


def patch(state: AgentState, **changes) -> AgentState:
    """Return a new state with the given keys overwritten."""
    return {**state, **changes}


class BaseAgent(Observable):
    """Single composed agent: orchestrates an LLMClient, MCPClient, and AgentMemory.

    Subclasses set `system_prompt` and `tool_names`, then either override `run()` or
    call `answer_with_tool()` from inside their own `run()`.
    """

    system_prompt: str = ""

    # None  → load all permitted MCP tools (default for generic agents).
    # []    → skip MCP connection entirely (hardcoded agents that never call tools).
    # [...] → load only the named tools.
    tool_names: list[str] | None = None

    _traced_methods: tuple[str, ...] = ("run",)
    _component_kind: str = "agent"
    _span_type: str = SpanType.AGENT

    def __init__(self) -> None:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        self.llm_client = LLMClient(llm, observer=self)
        self.mcp_client = MCPClient(observer=self)
        self.memory = AgentMemory()
        self.tools: list[BaseTool] = []
        if os.getenv("MCP_SERVER_URL"):
            self._load_mcp_from_env()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _increment(self, state: AgentState) -> AgentState:
        return patch(state, iteration_count=state.get("iteration_count", 0) + 1)

    def append_response(self, state: AgentState, text: str) -> AgentState:
        return patch(state, messages=[AIMessage(content=text)])

    def set_final_output(self, state: AgentState, text: str) -> AgentState:
        self.log_event(Events.FINAL_OUTPUT_SET, length=len(text) if text else 0)
        return patch(
            state,
            final_output=text,
            is_complete=True,
            messages=[AIMessage(content=text)],
        )

    def set_final_view(self, state: AgentState, text: str, view: dict) -> AgentState:
        self.log_event(
            Events.FINAL_OUTPUT_SET,
            length=len(text) if text else 0,
            view_type=view.get("type") if isinstance(view, dict) else None,
        )
        return patch(
            state,
            final_output=text,
            view=view,
            is_complete=True,
            messages=[AIMessage(content=text)],
        )

    def set_error(self, state: AgentState, msg: str) -> AgentState:
        self.log("error", Events.AGENT_ERROR_SET, agent=type(self).__name__, error=msg)
        return patch(state, error=msg, is_complete=True)

    def _append_trace(self, state: AgentState, **kwargs) -> AgentState:
        cls_name = type(self).__name__
        self.log_event(f"{cls_name}.trace", **{k: str(v) for k, v in kwargs.items()})
        entry = f"[{cls_name}] " + ", ".join(f"{k}={v}" for k, v in kwargs.items())
        existing = list(state.get("short_term_memory") or [])
        return patch(state, short_term_memory=existing + [entry])

    # ------------------------------------------------------------------
    # LLM / message helpers (used by subclasses with custom run())
    # ------------------------------------------------------------------

    def format_messages(self, state: AgentState) -> list[BaseMessage]:
        raw: list[BaseMessage] = state.get("messages") or []
        trimmed = self.memory.trim(raw)
        self.log_event(
            Events.FORMAT_MESSAGES,
            input_message_count=len(raw),
            trimmed_message_count=len(trimmed),
        )
        facts = state.get("long_term_memory_keys") or []
        return LLMClient.build_messages(
            self.system_prompt, trimmed, self.memory.facts_block(facts)
        )

    def call_llm(self, messages: list[BaseMessage]) -> str:
        return self.llm_client.call(messages)

    # ------------------------------------------------------------------
    # MCP loading + single-tool invocation
    # ------------------------------------------------------------------

    def _load_mcp_from_env(self) -> None:
        # Empty list means the subclass explicitly opted out — skip the connection.
        if self.tool_names is not None and not self.tool_names:
            return
        config = self.mcp_client.build_config_from_env()
        if not config:
            return
        try:
            self._run_async(self._async_load_mcp_tools(config))
            self.log(
                "info",
                Events.MCP_TOOLS_LOADED,
                agent=type(self).__name__,
                count=len(self.tools),
            )
        except Exception as e:
            self.log(
                "error",
                Events.MCP_LOAD_FAILED,
                agent=type(self).__name__,
                error=str(e),
                exc_info=True,
            )

    async def _async_load_mcp_tools(self, config) -> None:
        self.tools = await self.mcp_client.load_tools(config, self.tool_names)
        self.llm_client.bind_tools(self.tools)

    # ------------------------------------------------------------------
    # Agent-to-agent (A2A) — discover a peer via the Registry, message it
    # ------------------------------------------------------------------
    def call_peer(self, agent_id: str, args: dict, context: dict | None = None, *, sla_ms: int = 10000) -> str:
        """Delegate to a peer agent over A2A, discovered through the Registry.

        Returns the peer's text reply. Lets an agent fan work out to another
        agent mid-run (the "agents talk to each other" half of A2A Hybrid)
        without the two ever importing each other — discovery stays centralized
        in the Registry, transport is JSON-RPC ``message/send``.
        """
        from a2a.client import A2AClient
        from a2a.types import get_text

        reply = self._run_async(A2AClient().send(agent_id, args, context or {}, sla_ms=sla_ms))
        return get_text(reply)

    def call_mcp_tool(self, name: str, args: dict) -> str:
        tool = next((t for t in self.tools if t.name == name), None)
        if tool is None:
            raise LookupError(f"MCP tool '{name}' not available")
        with mlflow.start_span(name=f"mcp.{name}", span_type=SpanType.TOOL) as span:
            span.set_inputs({"tool": name, "args": args})
            raw = self._run_async(tool.ainvoke(args))
            report = MCPClient.unwrap_result(raw)
            span.set_outputs({"result": report})
            span.set_attribute("mcp.tool", name)
        self.log_event(Events.MCP_TOOL_CALL, tool=name, args=str(args), result=report[:200])
        return report

    def answer_with(
        self,
        state: AgentState,
        work: Callable[[], "str | tuple[str, dict | None]"],
        **trace_kwargs,
    ) -> AgentState:
        """Run a unit of agent work and capture its outcome on state.

        `work` is a zero-arg callable that returns either a plain text reply
        or a `(text, view)` tuple where `view` is a structured dict for the
        frontend renderer. Handles increment, exception capture, logging,
        tracing, and the final state mutation — subclasses just declare what
        to do, not the bookkeeping around it.
        """
        updated = self._increment(state)
        try:
            result = work()
        except LookupError as e:
            return self.set_error(updated, str(e))
        except Exception as e:
            self.log(
                "error",
                Events.AGENT_RUN_FAILED,
                agent=type(self).__name__,
                error=str(e),
                exc_info=True,
            )
            return self.set_error(updated, str(e))

        if isinstance(result, tuple):
            text, view = result
        else:
            text, view = result, None

        updated = self._append_trace(updated, **trace_kwargs)
        if view is not None:
            return self.set_final_view(updated, text, view)
        return self.set_final_output(updated, text)

    def answer_with_tool(
        self,
        state: AgentState,
        tool_name: str,
        args: dict,
        format_text: Callable[[str], str],
        **trace_kwargs,
    ) -> AgentState:
        """Shorthand for the single-MCP-tool case: call one tool, format its result."""
        def work() -> str:
            return format_text(self.call_mcp_tool(tool_name, args))

        return self.answer_with(state, work, source=f"mcp:{tool_name}", **trace_kwargs)

    # ------------------------------------------------------------------
    # Main agent loop
    # ------------------------------------------------------------------

    def run(self, state: AgentState) -> AgentState:
        state = self._increment(state)
        messages = self.format_messages(state)

        if not self.tools:
            return self.set_final_output(state, self.call_llm(messages))

        return self._run_tool_loop(state, messages)

    def _run_tool_loop(
        self,
        state: AgentState,
        messages: list[BaseMessage],
    ) -> AgentState:
        max_iters = state.get("max_iterations") or 10
        total_tool_calls = 0

        for iteration in range(max_iters):
            response = self.llm_client.invoke(messages)

            if not response.tool_calls:
                self._log_loop_end(iteration + 1, total_tool_calls, len(messages) + 1)
                return self.set_final_output(state, response.content)

            messages, state = self._step_tools(response, messages, state, iteration + 1)
            total_tool_calls += len(response.tool_calls)

        self._log_loop_end(max_iters, total_tool_calls, exceeded=True)
        return self.set_error(
            state, f"exceeded max_iterations ({max_iters}) without a final answer"
        )

    def _step_tools(
        self,
        response: AIMessage,
        messages: list[BaseMessage],
        state: AgentState,
        iteration: int,
    ) -> tuple[list[BaseMessage], AgentState]:
        self._log_tool_calls(iteration, response.tool_calls)
        messages.append(response)
        tool_messages = asyncio.run(self.llm_client.execute_tool_calls(response.tool_calls))
        messages.extend(tool_messages)
        self._log_tool_results(iteration, response.tool_calls, tool_messages)
        state = self._append_trace(state, tool_calls=len(response.tool_calls))
        return messages, state

    def _log_tool_calls(self, iteration: int, tool_calls: list[dict]) -> None:
        self.log_event(
            Events.LLM_TOOL_CALLS,
            iteration=iteration,
            count=len(tool_calls),
            calls=str([{"name": tc["name"], "args": tc["args"]} for tc in tool_calls]),
        )

    def _log_tool_results(
        self,
        iteration: int,
        tool_calls: list[dict],
        tool_messages: list[ToolMessage],
    ) -> None:
        self.log_event(
            Events.LLM_TOOL_RESULTS,
            iteration=iteration,
            results=str([
                {"name": tc["name"], "result": str(tm.content)[:200]}
                for tc, tm in zip(tool_calls, tool_messages)
            ]),
        )

    def _log_loop_end(
        self,
        iterations: int,
        total_tool_calls: int,
        final_message_count: int | None = None,
        exceeded: bool = False,
    ) -> None:
        attrs: dict = {"iterations": iterations, "total_tool_calls": total_tool_calls}
        if exceeded:
            attrs["exceeded_max_iters"] = True
        if final_message_count is not None:
            attrs["final_message_count"] = final_message_count
        self.log_event(Events.AGENT_SCRATCHPAD, **attrs)
