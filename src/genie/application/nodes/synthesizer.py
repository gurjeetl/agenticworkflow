"""Synthesizer node: composes the single user-facing answer from the blackboard.

The convergence point of every route. Has LLM-free fast paths (empty plan,
single structured view) and otherwise merges blackboard entries into prose, then
fires best-effort background write-backs (commits, embedding, fact extraction).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from genie.agents.base import BaseAgent, make_chat_model, patch
from genie.platform.config import get_settings
from genie.llm.client import LLMClient
from genie.memory.commit_store import get_commit_store
from genie.memory.facts_store import get_facts_store
from genie.memory.vector_store import get_vector_store
from genie.application.nodes._planner_dag import Plan
from genie.application.nodes._planner_parsing import extract_json
from genie.registry.registry_client import RegistryUnavailable, get_registry_client
from genie.application.state import AgentState

# Background executor for best-effort work that need not block the user's answer
# (durable fact extraction). Module-level so it's shared across synthesizer runs.
_BG = ThreadPoolExecutor(max_workers=2, thread_name_prefix="synth-bg")


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

    _FACTS_PROMPT = (
        "You extract durable facts from an assistant's answer. Return ONLY JSON of the form:\n"
        '{"facts":[{"key":"<short_snake_case>","value":"<concise string>","scope":"global|session"}]}\n'
        "- GLOBAL: stable facts about the user or the world that stay true across ALL future "
        "conversations (the user's name, their home city, a place's coordinates, a standing "
        "preference).\n"
        "- SESSION: facts that are only meaningful inside THIS conversation (a specific outage "
        "id the user is looking at, a one-off selection).\n"
        "- When unsure, choose \"session\".\n"
        "- Keys are lowercase snake_case, short and reusable. Omit anything that is not a "
        "confident, reusable fact. An empty list is fine.\n"
        "- Output JSON only — no markdown, no prose."
    )

    def __init__(self) -> None:
        """Set up the base agent, pointing the merge call at SYNTHESIZER_MODEL when configured (else the default model)."""
        super().__init__()
        # The synthesizer's merge call reads the whole blackboard (incl. large view
        # payloads), so it's the second-heaviest LLM call. Point it at a faster model
        # when configured; falls back to OPENAI_MODEL. Mirrors PLANNER_MODEL/ROUTER_MODEL.
        synth_model = get_settings().synthesizer_model
        if synth_model:
            self.llm_client = LLMClient(make_chat_model(synth_model), observer=self)

    def run(self, state: AgentState) -> AgentState:
        """Produce the final answer via a fast path or LLM merge, then write back this turn.

        Empty plan yields a clarification; a lone structured view passes through;
        otherwise the LLM merges entries, marking ``[PARTIAL]`` when tasks errored.
        """
        if get_settings().debug_break:
            breakpoint()  # opt-in: only fires when DEBUG_BREAK is set (see .vscode/launch.json)
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
            db_ops = self._writeback(state, blackboard, text)
            if view:
                return patch(self.set_final_view(updated, text, view), db_ops=db_ops)
            return patch(self.set_final_output(updated, text), db_ops=db_ops)

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

        db_ops = self._writeback(state, blackboard, text)
        return patch(self.set_final_output(updated, text), db_ops=db_ops)

    # ------------------------------------------------------------------
    @staticmethod
    def _render_blackboard(
        blackboard: dict[str, dict], per_entry_cap: int = 2500, total_cap: int = 8000
    ) -> str:
        """Serialize the blackboard for an LLM prompt, INCLUDING each entry's
        structured ``view`` — that's where list items / detail fields the user
        actually asked about live (e.g. the outage agent puts the 5 outages in
        ``view.items``; ``text`` is only a headline). Each entry is size-bounded so
        one large detail view can't crowd out the others, and the whole thing is
        capped to protect the token budget."""
        parts: list[str] = []
        for tid, entry in blackboard.items():
            if not isinstance(entry, dict):
                continue
            try:
                s = json.dumps(entry, default=str)
            except Exception:
                s = str(entry)
            if len(s) > per_entry_cap:
                s = s[:per_entry_cap] + "...(truncated)"
            parts.append(f'"{tid}": {s}')
        return ("{" + ", ".join(parts) + "}")[:total_cap]

    def _writeback(self, state: AgentState, blackboard: dict[str, dict], text: str) -> list[dict]:
        """Persist this turn and build the tracer op log.

        Two real writes: durable commits to MongoDB (``agent_commits``) and a
        semantic embedding of the final answer to Milvus (``long_term_memory``).
        Returns the per-step ``db_ops`` records for the Live DB State panel.
        """
        committed = self._commit_persistable(state, blackboard)
        db_ops: list[dict] = [
            {
                "store": "mongodb",
                "op": "write",
                "detail": f"commit {tid}",
                "code": "agent_commits.insertOne({...})",
                "enabled": True,
            }
            for tid in committed
        ]
        if not committed:
            db_ops.append({
                "store": "mongodb",
                "op": "write",
                "detail": "no persistable fields this turn",
                "enabled": True,
            })

        # Semantic write-back: embed the final answer for future recall. The embed
        # call hits an embedding model and its result is only read on FUTURE turns,
        # so run it in the background to keep it off the time-to-answer path.
        vector_store = get_vector_store()
        summary = (text or "").strip()
        embed_async = bool(summary) and vector_store.enabled
        if embed_async:
            _BG.submit(vector_store.add, state.get("thread_id") or "", summary[:1000])
            detail = "embed final answer → long_term_memory (async)"
        elif not vector_store.enabled:
            detail = "Milvus disabled"
        else:
            detail = "embed skipped (no answer text)"
        db_ops.append({
            "store": "milvus",
            "op": "write",
            "detail": detail,
            "code": "long_term_memory.insert(embed(answer))  # background",
            "enabled": vector_store.enabled,
        })

        # LLM-extract durable facts from the final answer → agent_facts. This is an
        # extra LLM call whose result is only read on the NEXT turn, so run it in the
        # background to keep it off the time-to-answer path. The method is best-effort
        # (catches its own errors) and its Mongo upserts are thread-safe. We mirror its
        # partial/empty gate here so the tracer panel stays honest.
        if state.get("partial") or not (text or "").strip():
            db_ops.append({
                "store": "mongodb",
                "op": "write",
                "detail": "facts extract skipped (partial)" if state.get("partial") else "facts extract skipped (no answer)",
                "enabled": True,
            })
        else:
            _BG.submit(self._extract_and_store_facts, state, blackboard, text)
            db_ops.append({
                "store": "mongodb",
                "op": "write",
                "detail": "facts extraction dispatched (async)",
                "code": "agent_facts.updateOne(...)  # background",
                "enabled": True,
            })
        self.log_event("synthesizer.writeback", commits=len(committed), milvus_async=embed_async)
        return db_ops

    def _extract_and_store_facts(self, state: AgentState, blackboard: dict[str, dict], text: str) -> dict:
        """Ask the LLM to pull reusable facts from the final answer and upsert them
        into ``agent_facts`` (global = stable across sessions; session = this thread
        only). Best-effort: any failure logs and returns a benign db_op — synthesis
        never crashes on it. Returns one tracer op record for the Live DB State panel.
        """
        # Gate: only extract from a real, complete, successful answer.
        if state.get("partial") or not (text or "").strip():
            return {
                "store": "mongodb",
                "op": "write",
                "detail": "facts extract skipped (partial)" if state.get("partial") else "facts extract skipped (no answer)",
                "enabled": True,
            }
        try:
            messages = [
                SystemMessage(content=self._FACTS_PROMPT),
                HumanMessage(
                    content=(
                        f"USER REQUEST:\n{state.get('user_input') or ''}\n\n"
                        f"FINAL ANSWER:\n{text}\n\n"
                        f"BLACKBOARD (JSON):\n{self._render_blackboard(blackboard)}\n\n"
                        "Extract the facts now."
                    )
                ),
            ]
            raw = self.call_llm(messages)
            parsed = extract_json(raw) or {}
            facts = self._validate_facts(parsed.get("facts"))
            store = get_facts_store()
            run_id = state.get("run_id") or ""
            thread_id = state.get("thread_id") or ""
            for f in facts:
                store.upsert(
                    scope=f["scope"],
                    key=f["key"],
                    value=f["value"],
                    thread_id=thread_id,
                    run_id=run_id,
                )
            if facts:
                return {
                    "store": "mongodb",
                    "op": "write",
                    "detail": f"extract {len(facts)} fact(s) → agent_facts",
                    "code": "agent_facts.updateOne({_id}, {...}, upsert=true)",
                    "enabled": True,
                    "hits": [f"{f['scope']}:{f['key']}={f['value']}"[:80] for f in facts],
                }
            return {"store": "mongodb", "op": "write", "detail": "no facts extracted", "enabled": True}
        except Exception as e:
            self.log("warning", "synthesizer.facts_extract_failed", error=str(e))
            return {"store": "mongodb", "op": "write", "detail": "facts extract failed", "enabled": True}

    @staticmethod
    def _validate_facts(raw: Any) -> list[dict]:
        """Coerce the LLM's facts array into clean {key, value, scope} dicts.
        Drops malformed entries; defaults unknown scope to 'session' (smaller blast
        radius); caps to 10 facts and 500-char values."""
        if not isinstance(raw, list):
            return []
        out: list[dict] = []
        for item in raw[:10]:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            value = item.get("value")
            if not isinstance(key, str) or not key.strip():
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            scope = item.get("scope")
            scope = scope if scope in ("global", "session") else "session"
            out.append({"key": key.strip(), "value": value.strip()[:500], "scope": scope})
        return out

    def _commit_persistable(self, state: AgentState, blackboard: dict[str, dict]) -> list[str]:
        """Persist fields whose schema marks persist=true. Returns committed task ids."""
        plan = Plan(**(state.get("plan") or {}))
        by_id = plan.by_id()
        store = get_commit_store()
        committed: list[str] = []
        if not store.enabled:
            return committed
        run_id = state.get("run_id") or ""
        thread_id = state.get("thread_id") or ""
        try:
            metas = {m.agent_id: m for m in get_registry_client().list_active()}
        except RegistryUnavailable as e:
            # Persistence is best-effort; without schemas we simply skip it.
            self.log("warning", "synthesizer.registry_unavailable", error=str(e))
            return committed
        for task_id, entry in blackboard.items():
            if not isinstance(entry, dict) or "error" in entry:
                continue
            subtask = by_id.get(task_id)
            if subtask is None:
                continue
            meta = metas.get(subtask.agent_id)
            if meta is None:
                continue
            payload: dict[str, Any] = {
                name: entry.get(name) for name, spec in meta.output_schema.items() if spec.persist
            }
            if not any(v is not None for v in payload.values()):
                continue
            try:
                store.commit(
                    run_id=run_id,
                    thread_id=thread_id,
                    agent_id=subtask.agent_id,
                    agent_version=subtask.agent_version,
                    task_id=task_id,
                    payload=payload,
                )
                committed.append(task_id)
            except Exception as e:
                self.log("warning", "synthesizer.commit_failed", task_id=task_id, error=str(e))
        return committed
