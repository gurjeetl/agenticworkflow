"""BaseAgent: the inheritance base every application agent subclasses.

Composes an LLMClient, MCPClient, and AgentMemory into a single agent, and
provides the shared run loop, state helpers, and MCP/A2A plumbing so subclasses
only declare their prompt, tools, and any custom work.
"""
import asyncio
from typing import Callable

import mlflow
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from mlflow.entities import SpanType

from genie.agents.memory import AgentMemory
from genie.platform.config import get_settings
from genie.platform.events import Events
from genie.llm.client import LLMClient
from genie.mcp.client import MCPClient
from genie.observability import Observable
from genie.application.state import AgentState

load_dotenv()


def patch(state: AgentState, **changes) -> AgentState:
    """Return a new state with the given keys overwritten."""
    return {**state, **changes}


def make_chat_model(model: str | None = None) -> ChatOpenAI:
    """Build a ChatOpenAI from the central Settings, with an optional model override.

    When a named LLM service is selected (``llm_services.default`` in YAML), the
    client points at that self-hosted endpoint; ``model`` then overrides only the
    model name while reusing the same endpoint. Otherwise it falls back to the flat
    OPENAI_MODEL / OPENAI_API_KEY / OPENAI_BASE_URL config.

    Set OPENAI_TEMPERATURE=0 to make the routing / planning / synthesis calls
    deterministic — that cuts run-to-run path variance (e.g. a prompt fast-pathing
    one run and full-planning the next) and the spurious re-plans it triggers. When
    unset, the provider default (~1.0) is used.
    """
    settings = get_settings()
    svc = settings.llm_services.active()
    if svc is not None:
        temp = svc.temperature
        if temp is None and settings.openai_temperature not in (None, ""):
            temp = float(settings.openai_temperature)
        kwargs: dict = {"temperature": temp} if temp is not None else {}
        return ChatOpenAI(
            model=model or svc.model_name,
            api_key=svc.api_key or "EMPTY",   # open endpoint → non-empty placeholder
            base_url=svc.base_url,            # e.g. http://genieapps4.dev.oati.local:8033/v1
            **kwargs,
        )
    kwargs = {}
    if settings.openai_temperature not in (None, ""):
        kwargs["temperature"] = float(settings.openai_temperature)
    return ChatOpenAI(
        model=model or settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        **kwargs,
    )


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
        """Wire up the LLM, MCP, and memory collaborators; load any env-configured MCP tools."""
        self.llm_client = LLMClient(make_chat_model(), observer=self)
        self.mcp_client = MCPClient(observer=self)
        self.memory = AgentMemory()
        self.tools: list[BaseTool] = []
        _settings = get_settings()
        if _settings.mcp_server_url or _settings.mcp_services:
            self._load_mcp_from_env()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _increment(self, state: AgentState) -> AgentState:
        """Bump the per-run iteration counter (used for loop/limit accounting)."""
        return patch(state, iteration_count=state.get("iteration_count", 0) + 1)

    def append_response(self, state: AgentState, text: str) -> AgentState:
        """Append an assistant message without marking the task complete."""
        return patch(state, messages=[AIMessage(content=text)])

    def set_final_output(self, state: AgentState, text: str) -> AgentState:
        """Record the agent's final text answer and flag the task complete."""
        self.log_event(Events.FINAL_OUTPUT_SET, length=len(text) if text else 0)
        return patch(
            state,
            final_output=text,
            is_complete=True,
            messages=[AIMessage(content=text)],
        )

    def set_final_view(self, state: AgentState, text: str, view: dict) -> AgentState:
        """Final answer plus a structured ``view`` dict for the frontend renderer."""
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
        """Record a terminal error and flag the task complete (no final output)."""
        self.log("error", Events.AGENT_ERROR_SET, agent=type(self).__name__, error=msg)
        return patch(state, error=msg, is_complete=True)

    def _append_trace(self, state: AgentState, **kwargs) -> AgentState:
        """Append a human-readable trace line to short-term memory and log it."""
        cls_name = type(self).__name__
        self.log_event(f"{cls_name}.trace", **{k: str(v) for k, v in kwargs.items()})
        entry = f"[{cls_name}] " + ", ".join(f"{k}={v}" for k, v in kwargs.items())
        existing = list(state.get("short_term_memory") or [])
        return patch(state, short_term_memory=existing + [entry])

    # ------------------------------------------------------------------
    # LLM / message helpers (used by subclasses with custom run())
    # ------------------------------------------------------------------

    def format_messages(self, state: AgentState) -> list[BaseMessage]:
        """Trim the message window and prepend system prompt + known facts for the LLM."""
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
        """One-shot LLM call returning plain text (no tool loop)."""
        return self.llm_client.call(messages)

    # ------------------------------------------------------------------
    # MCP loading + single-tool invocation
    # ------------------------------------------------------------------

    def _load_mcp_from_env(self) -> None:
        """Connect to the env-configured MCP server(s) and bind the agent's tools.

        Best-effort: failures are logged but never raised, so a missing/unreachable
        MCP server degrades the agent to LLM-only rather than failing construction.
        """
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
        """Load the permitted MCP tools and bind them to the LLM client."""
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
        from genie.a2a.client import A2AClient
        from genie.a2a.types import get_text

        reply = self._run_async(A2AClient().send(agent_id, args, context or {}, sla_ms=sla_ms))
        return get_text(reply)

    def call_mcp_tool(self, name: str, args: dict) -> str:
        """Invoke a single named MCP tool synchronously and return its text result.

        Wraps the call in an mlflow TOOL span and unwraps the MCP content blocks
        to a flat string. Raises ``LookupError`` if the tool was not loaded.
        """
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
            """Call the single tool and run its raw result through ``format_text``."""
            return format_text(self.call_mcp_tool(tool_name, args))

        return self.answer_with(state, work, source=f"mcp:{tool_name}", **trace_kwargs)

    # ------------------------------------------------------------------
    # Main agent loop
    # ------------------------------------------------------------------

    def run(self, state: AgentState) -> AgentState:
        """Default entry point: one LLM call when tool-less, else the tool loop.

        Subclasses with custom behavior override this and typically call
        ``answer_with`` / ``answer_with_tool`` instead.
        """
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
        """Iterate LLM↔tool calls until the model answers without a tool call.

        Stops with an error once ``max_iterations`` is exhausted, guarding against
        a model that keeps calling tools forever.
        """
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
        """Execute one round of the model's tool calls and append the results.

        Returns the extended message list (assistant turn + tool replies) and the
        state with a trace entry, ready to feed back into the next LLM invocation.
        """
        self._log_tool_calls(iteration, response.tool_calls)
        messages.append(response)
        tool_messages = asyncio.run(self.llm_client.execute_tool_calls(response.tool_calls))
        messages.extend(tool_messages)
        self._log_tool_results(iteration, response.tool_calls, tool_messages)
        state = self._append_trace(state, tool_calls=len(response.tool_calls))
        return messages, state

    def _log_tool_calls(self, iteration: int, tool_calls: list[dict]) -> None:
        """Emit a trace event naming the tools the model asked to call this iteration."""
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
        """Emit a trace event with each tool's (truncated) result for this iteration."""
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
        """Emit the loop-end trace event (iteration/tool totals, or the limit-exceeded marker)."""
        attrs: dict = {"iterations": iterations, "total_tool_calls": total_tool_calls}
        if exceeded:
            attrs["exceeded_max_iters"] = True
        if final_message_count is not None:
            attrs["final_message_count"] = final_message_count
        self.log_event(Events.AGENT_SCRATCHPAD, **attrs)
