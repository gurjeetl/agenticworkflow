"""Front-of-graph Router: cheap, registry-aware intent triage.

Decides one of three routes before the expensive Planner runs:
  - ``fast``     — the request maps to exactly one agent with fillable args →
                   build a one-task plan + waves and jump straight to the Executor.
  - ``chitchat`` — greeting / thanks / meta, no agent needed → straight to the
                   Synthesizer (its empty-plan path returns a clarification).
  - ``plan``     — anything ambiguous or multi-intent → the full Planner.

It is a *fast mirror* of the Planner: it reads the same registry capability menu
(via ``planner.parsing``) and **fails open to ``plan``** on any doubt, registry
outage, or LLM/parse failure — so it can only ever speed things up, never reduce
what the system can answer.
"""
from __future__ import annotations

import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from baseagent.base_agent import BaseAgent, patch
from baseagent.llm_client import LLMClient
from planner.dag import Plan, Subtask
from planner.parsing import extract_json, normalize_agent_id, render_capability_menu
from registry.registry_client import RegistryUnavailable, get_registry_client
from router.intent_classifier import get_intent_classifier
from state import AgentState

_ROUTER_SCHEMA_HINT = (
    'Respond ONLY with valid JSON in this exact shape:\n'
    '{"route":"fast|chitchat|plan","agent_id":"<one agent_id or null>",'
    '"args":{...},"confidence":0.0}\n'
    "No extra text, no markdown fences, no explanation — just the JSON."
)

# Cheap, pre-LLM signal that a prompt is clearly multi-intent. Such prompts always
# fall through to the planner anyway, so matching one lets us skip the router's LLM
# call entirely. Conservative additive connectors only: a miss just means we pay the
# router call we'd have paid; a false hit only costs a single-agent prompt its fast
# path (it still gets answered, via the planner). Override with ROUTER_MULTI_INTENT_PATTERN.
_DEFAULT_MULTI_INTENT_PATTERN = r"(?i)\b(also|as well as|and also|additionally|moreover)\b|;"


class RouterAgent(BaseAgent):
    tool_names: list[str] = []

    def __init__(self) -> None:
        super().__init__()
        self._registry = get_registry_client()
        # Use a cheaper/faster model for routing when configured; the win of the
        # fast path is realised only if the triage call is cheap.
        router_model = os.getenv("ROUTER_MODEL")
        if router_model:
            llm = ChatOpenAI(
                model=router_model,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL") or None,
            )
            self.llm_client = LLMClient(llm, observer=self)
        self._min_confidence = float(os.getenv("ROUTER_MIN_CONFIDENCE", "0.7"))
        self._multi_intent_re = re.compile(
            os.getenv("ROUTER_MULTI_INTENT_PATTERN") or _DEFAULT_MULTI_INTENT_PATTERN
        )

    # ------------------------------------------------------------------
    def _build_system_prompt(self, metas) -> str:
        menu = render_capability_menu(metas)
        return (
            "You are a fast intent ROUTER sitting in front of a planner. Pick the "
            "cheapest correct route for the user's message. Choose exactly ONE:\n\n"
            "- \"fast\": the message maps to EXACTLY ONE agent below and you can fill "
            "its required inputs (marked *). Put the agent_id and args.\n"
            "- \"chitchat\": greeting, thanks, small talk, or a meta question "
            "(\"what can you do?\") that needs NO agent. agent_id=null, args={}.\n"
            "- \"plan\": ANYTHING else — multiple agents needed, ambiguous, missing "
            "required info, or you are unsure. THIS IS THE SAFE DEFAULT.\n\n"
            "REGISTERED AGENTS:\n"
            f"{menu}\n\n"
            "Rules:\n"
            "- When in doubt, choose \"plan\". Only choose \"fast\" when one agent "
            "clearly and solely satisfies the request.\n"
            "- If the request needs two or more agents, choose \"plan\".\n"
            "- confidence is your 0.0-1.0 certainty in a \"fast\" match.\n"
            "- City names go in args as lowercase strings.\n\n"
            "Examples:\n"
            'User: "What\'s the weather in Paris?" → {"route":"fast","agent_id":"weather","args":{"location":"paris"},"confidence":0.95}\n'
            'User: "hi there" → {"route":"chitchat","agent_id":null,"args":{},"confidence":0.0}\n'
            'User: "weather in Tokyo and the top outages" → {"route":"plan","agent_id":null,"args":{},"confidence":0.0}\n\n'
            f"{_ROUTER_SCHEMA_HINT}"
        )

    # ------------------------------------------------------------------
    def run(self, state: AgentState) -> AgentState:
        if os.getenv("DEBUG_BREAK"):
            breakpoint()  # opt-in: only fires when DEBUG_BREAK is set (see .vscode/launch.json)
        updated = self._increment(state)
        user_msg = state.get("user_input") or ""

        try:
            metas = self._registry.list_active()
        except RegistryUnavailable as e:
            self.log("warning", "router.registry_unavailable", error=str(e))
            return self._route_plan(updated, reason="registry_unavailable")

        # Clearly multi-intent prompts fall through to the planner regardless, so skip
        # the router's LLM call for them. Two cheap, local signals (either fires):
        #   - regex: explicit additive phrasing ("...outages. ALSO the weather")
        #   - classifier: prompt activates >= 2 distinct agents by embedding similarity
        #     (catches implicit "weather in tokyo and the top outages"). Fails open.
        if user_msg:
            reason = self._multi_intent_reason(user_msg, metas)
            if reason:
                return self._route_plan(updated, reason=reason)

        prompt = self._build_system_prompt(metas)
        try:
            raw = self.call_llm([SystemMessage(content=prompt), HumanMessage(content=user_msg)])
        except Exception as e:
            self.log("warning", "router.llm_failed", error=str(e))
            return self._route_plan(updated, reason="llm_failed")

        parsed = extract_json(raw) or {}
        route = str(parsed.get("route") or "plan").strip().lower()

        if route == "chitchat":
            return self._route_chitchat(updated)
        if route == "fast":
            fast = self._try_fast(updated, parsed, metas)
            if fast is not None:
                return fast
        return self._route_plan(updated, reason="default")

    # ------------------------------------------------------------------
    def _multi_intent_reason(self, user_msg: str, metas) -> str | None:
        """Return a route reason if the prompt is clearly multi-intent, else None.

        Regex first (free); then the local embedding classifier (counts activated
        agents). Either signal is sufficient. The classifier fails open (count 0).
        """
        if self._multi_intent_re.search(user_msg):
            return "multi_intent_regex"
        if get_intent_classifier().is_multi_intent(user_msg, metas):
            return "multi_intent_classifier"
        return None

    # ------------------------------------------------------------------
    def _try_fast(self, state: AgentState, parsed: dict, metas) -> AgentState | None:
        """Validate a 'fast' decision; return routed state, or None to downgrade."""
        by_id = {m.agent_id: m for m in metas}
        agent_id = normalize_agent_id(parsed.get("agent_id"), set(by_id))
        meta = by_id.get(agent_id) if agent_id else None
        if meta is None:
            self.log("warning", "router.fast_unknown_agent", raw=str(parsed.get("agent_id")))
            return None

        try:
            confidence = float(parsed.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < self._min_confidence:
            self.log_event("router.fast_low_confidence", agent_id=agent_id, confidence=confidence)
            return None

        args = parsed.get("args") or {}
        ok, err = meta.validate_args(args)
        if not ok:
            self.log_event("router.fast_invalid_args", agent_id=agent_id, error=err)
            return None

        subtask = Subtask(
            id="t1",
            agent_id=meta.agent_id,
            agent_version=meta.version,
            args=args,
            depends_on=[],
            sla_ms=meta.sla_ms,
        )
        plan = Plan(subtasks=[subtask])
        self.log_event("router.decision", route="fast", agent_id=meta.agent_id, confidence=confidence)
        return patch(
            state,
            route="fast",
            plan=plan.model_dump(),
            agent_versions={"t1": meta.version},
            waves=[["t1"]],
            blackboard={},
            blackboard_snapshot=None,
        )

    def _route_chitchat(self, state: AgentState) -> AgentState:
        self.log_event("router.decision", route="chitchat")
        return patch(
            state,
            route="chitchat",
            plan={"subtasks": []},
            agent_versions={},
            blackboard={},
        )

    def _route_plan(self, state: AgentState, *, reason: str = "default") -> AgentState:
        self.log_event("router.decision", route="plan", reason=reason)
        return patch(state, route="plan")
