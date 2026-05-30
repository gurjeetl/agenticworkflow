from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from baseagent.base_agent import BaseAgent
from memory.postgres_store import get_postgres_store
from planner.dag import Plan
from registry import AGENT_REGISTRY
from state import AgentState


class SynthesizerAgent(BaseAgent):
    """Reads the blackboard and composes one user-facing answer.

    Two fast paths shortcut the LLM:
      - empty plan → friendly clarification.
      - exactly one task with a structured view → pass that view through unchanged
        (preserves the existing /chat {response, view} contract).
    Otherwise the LLM merges the blackboard entries into prose.
    """

    tool_names: list[str] = []
    system_prompt = (
        "You are a synthesis agent. You will receive a JSON blackboard whose keys are "
        "task ids and whose values are agent outputs (or {\"error\": ...} entries). "
        "Compose one concise, helpful answer to the user's original request by merging "
        "the successful outputs. For any blackboard entry that contains an error, mark "
        "that section [PARTIAL] in the final answer. Do not invent facts. Do not include "
        "raw JSON in the output."
    )

    def run(self, state: AgentState) -> AgentState:
        updated = self._increment(state)
        blackboard: dict[str, dict] = state.get("blackboard") or {}
        plan = Plan(**(state.get("plan") or {}))

        # Empty plan → clarification.
        if not plan.subtasks:
            return self.set_final_output(
                updated,
                "I can help with weather or grid outages. Could you tell me what you need?",
            )

        # Single task with a structured view → pass through (preserves existing UX).
        successful = [
            (tid, entry) for tid, entry in blackboard.items() if isinstance(entry, dict) and "error" not in entry
        ]
        if len(plan.subtasks) == 1 and len(successful) == 1:
            tid, entry = successful[0]
            view = entry.get("view")
            text = entry.get("text") or ""
            if state.get("partial"):
                text = f"[PARTIAL] {text}".strip()
            self._commit_persistable(state, blackboard)
            if view:
                return self.set_final_view(updated, text, view)
            return self.set_final_output(updated, text)

        # Multi-task or no view — LLM-synthesize prose.
        user_input = state.get("user_input") or ""
        bb_for_prompt = self._render_blackboard(blackboard)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=(
                    f"USER REQUEST:\n{user_input}\n\n"
                    f"BLACKBOARD (JSON):\n{bb_for_prompt}\n\n"
                    "Compose the final answer now."
                )
            ),
        ]
        try:
            text = self.call_llm(messages)
        except Exception as e:
            self.log("error", "synthesizer.llm_failed", error=str(e))
            return self.set_error(updated, "Could not compose the final answer.")

        if state.get("partial") and "[PARTIAL]" not in text:
            text = f"[PARTIAL] {text}"

        self._commit_persistable(state, blackboard)
        return self.set_final_output(updated, text)

    # ------------------------------------------------------------------
    @staticmethod
    def _render_blackboard(blackboard: dict[str, dict]) -> str:
        # Strip 'view' from the prompt — it's frontend-only and would bloat tokens.
        slimmed = {}
        for tid, entry in blackboard.items():
            if not isinstance(entry, dict):
                continue
            slimmed[tid] = {k: v for k, v in entry.items() if k != "view"}
        try:
            return json.dumps(slimmed, default=str)[:4000]
        except Exception:
            return str(slimmed)[:4000]

    def _commit_persistable(self, state: AgentState, blackboard: dict[str, dict]) -> None:
        """Walk each blackboard entry and persist fields whose schema marks persist=true."""
        plan = Plan(**(state.get("plan") or {}))
        by_id = plan.by_id()
        store = get_postgres_store()
        if not store.enabled:
            return
        run_id = state.get("run_id") or ""
        thread_id = state.get("thread_id") or ""
        for task_id, entry in blackboard.items():
            if not isinstance(entry, dict) or "error" in entry:
                continue
            subtask = by_id.get(task_id)
            if subtask is None:
                continue
            meta_entry = AGENT_REGISTRY.get(subtask.agent_id)
            if meta_entry is None:
                continue
            meta, _cls = meta_entry
            payload: dict[str, Any] = {
                name: entry.get(name) for name, spec in meta.output_schema.items() if spec.persist
            }
            if not any(v is not None for v in payload.values()):
                continue
            try:
                self._run_async(
                    store.commit(
                        run_id=run_id,
                        thread_id=thread_id,
                        agent_id=subtask.agent_id,
                        agent_version=subtask.agent_version,
                        task_id=task_id,
                        payload=payload,
                    )
                )
            except Exception as e:
                self.log("warning", "synthesizer.commit_failed", task_id=task_id, error=str(e))
