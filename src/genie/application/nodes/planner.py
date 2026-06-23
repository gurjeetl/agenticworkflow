"""Planner node: turns the user request into a validated DAG of agent subtasks.

The heaviest LLM call in the pipeline. Builds a registry-derived capability menu
(plus semantic + structural memory recall and any re-plan context), prompts the
model for a subtask graph, then validates each subtask against the live registry.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from genie.agents.base import BaseAgent, make_chat_model, patch
from genie.platform.config import get_settings
from genie.platform.events import Events
from genie.llm.client import LLMClient
from genie.memory.facts_store import get_facts_store
from genie.memory.vector_store import get_vector_store
from genie.application.nodes._planner_dag import Plan, Subtask
from genie.application.nodes._planner_parsing import extract_json, normalize_agent_id, render_capability_menu
from genie.registry.registry_client import RegistryUnavailable, get_registry_client
from genie.application.state import AgentState

_PLAN_SCHEMA_HINT = (
    'Respond ONLY with valid JSON in this exact shape:\n'
    '{"subtasks":['
    '{"id":"t1","agent_id":"<one of the agents>","args":{...},"depends_on":[],"sla_ms":10000}'
    ']}\n'
    "No extra text, no markdown fences, no explanation — just the JSON."
)

# Upper bound on recalled facts injected into the planner prompt. The facts block
# is the DYNAMIC, un-cacheable part of the prompt and grows every turn as the
# session accumulates facts — so without a cap the planner prompt (and its latency)
# drifts upward over a long conversation. 40 is generous (normal turns recall a
# handful); it only clips pathological sessions. Override with PLANNER_MAX_FACTS.
_MAX_FACTS = get_settings().planner_max_facts


class PlannerAgent(BaseAgent):
    """Splits the user prompt into a DAG of subtasks, one per matched agent.

    No MCP tools — pure LLM with registry-derived menu in the prompt.
    """

    tool_names: list[str] = []

    def __init__(self) -> None:
        """Set up the base agent and Registry client; the agent-menu system prompt is built per-run."""
        super().__init__()
        # System prompt is built per-run so newly-discovered agents appear automatically.
        self.system_prompt = ""
        self._registry = get_registry_client()
        # The planner is the single heaviest LLM call (big agent-menu prompt + a
        # generated DAG). Point it at a faster model when configured; falls back to
        # OPENAI_MODEL. Mirrors ROUTER_MODEL.
        planner_model = get_settings().planner_model
        if planner_model:
            self.llm_client = LLMClient(make_chat_model(planner_model), observer=self)

    # ------------------------------------------------------------------
    @staticmethod
    def _recall_op(vector_store, recall: list[dict]) -> dict:
        """Build the tracer op record for the Milvus semantic-recall step."""
        if not vector_store.enabled:
            return {
                "store": "milvus",
                "op": "search",
                "detail": "semantic recall — Milvus disabled",
                "enabled": False,
            }
        return {
            "store": "milvus",
            "op": "search",
            "detail": f"semantic recall — {len(recall)} hit(s)",
            "code": "long_term_memory.search(embed(prompt), limit=5)",
            "enabled": True,
            "hits": [str(h.get("content", ""))[:80] for h in recall],
        }

    @staticmethod
    def _facts_op(facts: dict[str, str]) -> dict:
        """Build the tracer op record for the agent_facts structural-recall step."""
        return {
            "store": "mongodb",
            "op": "read",
            "detail": f"facts recall — {len(facts)} fact(s)",
            "code": "agent_facts.find({scope:global} | {scope:session, thread_id})",
            "enabled": True,
            "hits": [f"{k}: {v}"[:80] for k, v in facts.items()],
        }

    # ------------------------------------------------------------------
    def _build_system_prompt(
        self, state: AgentState, recall: list[dict] | None = None, facts: dict[str, str] | None = None
    ) -> str:
        """Assemble the planning prompt: capability menu + recall/facts/re-plan blocks.

        The menu is the cacheable part; recall, facts, and re-plan context are the
        dynamic blocks appended per turn.
        """
        menu = render_capability_menu(self._registry.list_active())
        recall_block = ""
        if recall:
            lines = "\n".join(f"- {str(h.get('content', '')).strip()}" for h in recall)
            recall_block = (
                "\n\nRELEVANT PAST CONTEXT (semantic recall from long-term memory — "
                "use only if it helps; do not invent facts):\n" + lines
            )
        facts_block = ""
        if facts:
            lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
            facts_block = (
                "\n\nKNOWN FACTS (structured recall from agent_facts — use only if it "
                "helps; do not invent facts):\n" + lines
            )
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
            "- CHAINING: to feed an earlier subtask's result into a later one, put a "
            "reference in the later subtask's args AND add <id> to its depends_on. Use "
            "${<id>.text} for the task's text output, or ${<id>.view.<path>} for a field of "
            "its structured view (see each agent's 'outputs' for the shape). The reference is "
            "replaced at run time. For 'the first/Nth one of a list', reference the real field "
            "(e.g. ${t1.view.items.0.id}) — do NOT guess a literal like outage_id 1. Chain only "
            "when the later task genuinely needs the earlier task's output.\n"
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
            'User: "Top 5 outages, then full details of the first one." (chained — reference the real id)\n'
            '→ {"subtasks":['
            '{"id":"t1","agent_id":"outage","args":{},"depends_on":[]},'
            '{"id":"t2","agent_id":"outage","args":{"outage_id":"${t1.view.items.0.id}"},"depends_on":["t1"]}'
            ']}\n\n'
            'User: "Look up outage 18645677, then have the docs assistant explain it." (chained)\n'
            '→ {"subtasks":['
            '{"id":"t1","agent_id":"outage","args":{"outage_id":18645677},"depends_on":[]},'
            '{"id":"t2","agent_id":"rag","args":{"query":"Explain this outage: ${t1.text}"},"depends_on":["t1"]}'
            ']}\n\n'
            "Output rules:\n"
            "- Use only agent_ids from the list above.\n"
            "- Give each subtask a stable id like 't1','t2'.\n"
            "- City names go in args as lowercase strings.\n"
            f"{recall_block}"
            f"{facts_block}"
            f"{replan_block}\n\n"
            f"{_PLAN_SCHEMA_HINT}"
        )

    # ------------------------------------------------------------------
    def _build_plan(self, parsed: dict) -> Plan:
        """Validate the model's raw subtasks into a Plan against the live registry.

        Drops subtasks naming an unknown agent or with invalid args (the run still
        proceeds with whatever validated), normalizing ids and stamping versions.
        """
        raw_subtasks: list[dict[str, Any]] = parsed.get("subtasks", []) or []
        metas = {m.agent_id: m for m in self._registry.list_active()}
        known_ids = set(metas)
        clean: list[Subtask] = []
        for st in raw_subtasks:
            raw_id = st.get("agent_id")
            agent_id = normalize_agent_id(raw_id, known_ids)
            meta = metas.get(agent_id) if agent_id else None
            if meta is None:
                self.log("warning", "planner.unknown_agent_id", raw=str(raw_id), normalized=str(agent_id))
                continue
            if agent_id != raw_id:
                self.log_event("planner.agent_id_normalized", raw=str(raw_id), resolved=agent_id)
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
        """Recall memory, prompt the LLM for a plan, validate it, and reset the blackboard.

        Sets an error on registry/parse failure; an empty plan routes to the
        Synthesizer for a clarification.
        """
        if get_settings().debug_break:
            breakpoint()  # opt-in: only fires when DEBUG_BREAK is set (see .vscode/launch.json)
        updated = self._increment(state)
        user_msg = state.get("user_input") or ""

        # Real semantic recall from Milvus long-term memory (no-ops when disabled).
        vector_store = get_vector_store()
        recall = vector_store.search(state.get("thread_id") or "", user_msg) if user_msg else []
        db_ops = [self._recall_op(vector_store, recall)]
        self.log_event("planner.semantic_recall", hits=len(recall), enabled=vector_store.enabled)

        # Structured recall from agent_facts (globals + this thread's session facts).
        # Cap the count so the prompt can't grow unbounded as a session accumulates
        # facts (keeps planner latency stable across a long conversation).
        facts = get_facts_store().query(state.get("thread_id") or "")
        if len(facts) > _MAX_FACTS:
            self.log_event("planner.facts_capped", total=len(facts), kept=_MAX_FACTS)
            facts = dict(list(facts.items())[:_MAX_FACTS])
        db_ops.append(self._facts_op(facts))
        self.log_event("planner.facts_recall", facts=len(facts))

        try:
            prompt = self._build_system_prompt(state, recall, facts)
        except RegistryUnavailable as e:
            self.log("error", "planner.registry_unavailable", error=str(e))
            return self.set_error(updated, "Agent registry is unavailable; cannot build a plan.")
        messages = [SystemMessage(content=prompt), HumanMessage(content=user_msg)]
        raw = self.call_llm(messages)
        updated = patch(updated, agent_scratchpad=raw)

        parsed = extract_json(raw)
        if parsed is None:
            self.log("error", "planner.parse_failed", raw=raw[:500])
            return self.set_error(updated, "Planner could not parse a plan from the model.")

        try:
            plan = self._build_plan(parsed)
        except RegistryUnavailable as e:
            self.log("error", "planner.registry_unavailable", error=str(e))
            return self.set_error(updated, "Agent registry is unavailable; cannot validate the plan.")
        if not plan.subtasks:
            self.log_event("planner.empty_plan")
            # Empty plan → Synthesizer will produce a clarification.
            return patch(
                updated,
                plan=plan.model_dump(),
                agent_versions={},
                blackboard={},
                db_ops=db_ops,
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
            db_ops=db_ops,
        )
