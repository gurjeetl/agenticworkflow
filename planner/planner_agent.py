from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from baseagent.base_agent import BaseAgent, patch
from baseagent.events import Events
from planner.dag import Plan, Subtask
from registry import AGENT_REGISTRY, list_active
from state import AgentState

_PLAN_SCHEMA_HINT = (
    'Respond ONLY with valid JSON in this exact shape:\n'
    '{"subtasks":['
    '{"id":"t1","agent_id":"<one of the agents>","args":{...},"depends_on":[],"sla_ms":10000}'
    ']}\n'
    "No extra text, no markdown fences, no explanation — just the JSON."
)


class PlannerAgent(BaseAgent):
    """Splits the user prompt into a DAG of subtasks, one per matched agent.

    No MCP tools — pure LLM with registry-derived menu in the prompt.
    """

    tool_names: list[str] = []

    def __init__(self) -> None:
        super().__init__()
        # System prompt is built per-run so newly-registered agents appear automatically.
        self.system_prompt = ""

    # ------------------------------------------------------------------
    def _render_capability_menu(self) -> str:
        lines = []
        for meta, _cls in list_active():
            inputs = ", ".join(
                f"{name}{'*' if spec.required else ''}:{spec.type}"
                for name, spec in meta.input_schema.items()
            ) or "(none)"
            tags = ", ".join(meta.capability_tags) or "(none)"
            lines.append(
                f'- agent_id: "{meta.agent_id}"   (use this exact string; the version below is INFO ONLY, do NOT include it)\n'
                f'    version: {meta.version}\n'
                f"    capability: {meta.description or '(no description)'}\n"
                f"    tags: {tags}\n"
                f"    inputs: {inputs}\n"
                f"    sla_ms: {meta.sla_ms}"
            )
        return "\n".join(lines) if lines else "(no agents registered)"

    def _build_system_prompt(self, state: AgentState) -> str:
        menu = self._render_capability_menu()
        replan_block = ""
        snapshot = state.get("blackboard_snapshot")
        reason = state.get("replan_reason")
        if snapshot or reason:
            replan_block = (
                "\n\nRE-PLAN CONTEXT (previous attempt's blackboard + reason):\n"
                f"reason: {reason or '(none)'}\n"
                f"snapshot: {json.dumps(snapshot, default=str)[:2000]}\n"
                "Adjust the plan to recover from the errors above."
            )
        return (
            "You are a planning agent. Look at the user's request and split it into "
            "one or more SUBTASKS, where each subtask is assigned to exactly one "
            "registered agent below. Match user intent to agent capability + tags.\n\n"
            "REGISTERED AGENTS:\n"
            f"{menu}\n\n"
            "How to match:\n"
            "- Read each agent's capability description AND tags. The tags are hints, "
            "not the full story — phrasing like 'show', 'list', 'tell me about', 'top N', "
            "'forecast', 'report' are common synonyms; match the agent that performs the "
            "underlying capability, even if the user uses a different word.\n"
            "- Required inputs are marked with an asterisk (*). Optional inputs may be "
            "omitted — when an agent works fine with empty args, pass {}.\n"
            "- depends_on=[] means a subtask can run independently. Populate depends_on "
            "ONLY when one task literally needs another task's output as input. "
            "Two unrelated requests run in parallel.\n"
            "- Only return an empty subtasks list when truly NO registered agent can "
            "address the request. If you can find a reasonable match, return that match.\n\n"
            "Examples:\n"
            'User: "What\'s the weather in Paris?"\n'
            '→ {"subtasks":[{"id":"t1","agent_id":"weather","args":{"location":"paris"},"depends_on":[]}]}\n\n'
            'User: "Show me the top 5 outages."\n'
            '→ {"subtasks":[{"id":"t1","agent_id":"outage","args":{},"depends_on":[]}]}\n\n'
            'User: "Tell me about outage 17299126."\n'
            '→ {"subtasks":[{"id":"t1","agent_id":"outage","args":{"outage_id":17299126},"depends_on":[]}]}\n\n'
            'User: "Weather in Tokyo and the top outages."\n'
            '→ {"subtasks":['
            '{"id":"t1","agent_id":"weather","args":{"location":"tokyo"},"depends_on":[]},'
            '{"id":"t2","agent_id":"outage","args":{},"depends_on":[]}'
            ']}\n\n'
            "Output rules:\n"
            "- Use only agent_ids from the list above.\n"
            "- Give each subtask a stable id like 't1','t2'.\n"
            "- City names go in args as lowercase strings.\n"
            f"{replan_block}\n\n"
            f"{_PLAN_SCHEMA_HINT}"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        """Find the first balanced JSON object in `raw` and parse it.

        Tolerant of LLM tics like trailing junk, an extra closing brace, or a
        markdown code fence — we walk the string tracking brace depth and string
        state so we stop exactly at the matching closer of the first object.
        """
        if not raw:
            return None
        start = raw.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        return None
        # Unbalanced — fall back to greedy parse as a last resort.
        try:
            return json.loads(raw[start:])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize_agent_id(raw_id: str | None) -> str | None:
        """Resolve common LLM stumbles to a real registry key.

        Handles: trailing version (` v1.0.0`), accidental quotes/whitespace,
        case differences. Returns the canonical agent_id if a match is found,
        else None.
        """
        if not raw_id or not isinstance(raw_id, str):
            return None
        cleaned = raw_id.strip().strip('"').strip("'").strip()
        # Drop trailing " v1.2.3" or "@1.2.3" version suffixes.
        cleaned = re.sub(r"[\s@]+v?\d+(?:\.\d+){0,3}\s*$", "", cleaned).strip()
        if cleaned in AGENT_REGISTRY:
            return cleaned
        # Case-insensitive fallback.
        lower_map = {k.lower(): k for k in AGENT_REGISTRY}
        return lower_map.get(cleaned.lower())

    def _build_plan(self, parsed: dict) -> Plan:
        raw_subtasks: list[dict[str, Any]] = parsed.get("subtasks", []) or []
        clean: list[Subtask] = []
        for st in raw_subtasks:
            raw_id = st.get("agent_id")
            agent_id = self._normalize_agent_id(raw_id)
            entry = AGENT_REGISTRY.get(agent_id) if agent_id else None
            if entry is None:
                self.log("warning", "planner.unknown_agent_id", raw=str(raw_id), normalized=str(agent_id))
                continue
            if agent_id != raw_id:
                self.log_event("planner.agent_id_normalized", raw=str(raw_id), resolved=agent_id)
            meta, _cls = entry
            args = st.get("args") or {}
            ok, err = meta.validate_args(args)
            if not ok:
                self.log("warning", "planner.invalid_args", agent_id=agent_id, error=err)
                continue
            clean.append(
                Subtask(
                    id=str(st.get("id") or f"t{len(clean) + 1}"),
                    agent_id=agent_id,
                    agent_version=meta.version,
                    args=args,
                    depends_on=list(st.get("depends_on") or []),
                    sla_ms=int(st.get("sla_ms") or meta.sla_ms),
                )
            )
        return Plan(subtasks=clean)

    # ------------------------------------------------------------------
    def run(self, state: AgentState) -> AgentState:
        updated = self._increment(state)
        prompt = self._build_system_prompt(state)
        user_msg = state.get("user_input") or ""
        messages = [SystemMessage(content=prompt), HumanMessage(content=user_msg)]
        raw = self.call_llm(messages)
        updated = patch(updated, agent_scratchpad=raw)

        parsed = self._extract_json(raw)
        if parsed is None:
            self.log("error", "planner.parse_failed", raw=raw[:500])
            return self.set_error(updated, "Planner could not parse a plan from the model.")

        plan = self._build_plan(parsed)
        if not plan.subtasks:
            self.log_event("planner.empty_plan")
            # Empty plan → Synthesizer will produce a clarification.
            return patch(
                updated,
                plan=plan.model_dump(),
                agent_versions={},
                blackboard={},
            )

        agent_versions = {t.id: t.agent_version for t in plan.subtasks}
        self.log_event(
            "planner.plan_built",
            count=len(plan.subtasks),
            agent_ids=str([t.agent_id for t in plan.subtasks]),
        )
        # Reset blackboard / snapshot for the fresh execution wave.
        return patch(
            updated,
            plan=plan.model_dump(),
            agent_versions=agent_versions,
            blackboard={},
            blackboard_snapshot=None,
            replan_reason=None,
        )
