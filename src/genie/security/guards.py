"""Two graph nodes that bracket the pipeline with LLM Guard scanning.

``InputGuard`` runs before the Router (scans the user prompt); ``OutputGuard``
runs after the Synthesizer (scans the final answer). Both are lightweight
``Observable`` nodes following the CompletionGate pattern — a traced
``run(state) -> state`` with no LLM and no DB I/O.

Blocking reuses the chitchat fast-path mechanism: set ``final_output`` +
``is_complete=True`` and let the conditional edge route straight to END.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage
from mlflow.entities import SpanType

from genie.agents.base import patch
from genie.observability import Observable
from genie.platform.config import get_settings
from genie.security.llm_guard import get_llm_guard
from genie.application.state import AgentState

_REFUSAL = (
    "I can't help with that request — it was flagged by our safety filter. "
    "Please rephrase and try again."
)


class InputGuard(Observable):
    """Scan the incoming user prompt; block high-risk content, redact PII/secrets."""

    _traced_methods: tuple[str, ...] = ("run",)
    _component_kind: str = "guard"
    _span_type: str = SpanType.CHAIN

    def run(self, state: AgentState) -> AgentState:
        """Scan ``user_input``; on a block short-circuit to a refusal, else continue
        downstream with the sanitized (PII/secret-redacted) prompt."""
        if get_settings().debug_break:
            breakpoint()  # opt-in: only fires when DEBUG_BREAK is set (see .vscode/launch.json)
        guard = get_llm_guard()
        text = state.get("user_input") or ""
        res = guard.scan_input(text)

        if not res["valid"]:
            self.log("warning", "input_guard.blocked", findings=res["findings"])
            return patch(
                state,
                guard_block={"stage": "input", "findings": res["findings"], "scores": res["scores"]},
                guard_input={"scanned": True, "blocked": True, "findings": res["findings"],
                             "scores": res["scores"], "redacted": False},
                final_output=_REFUSAL,
                is_complete=True,
                messages=[AIMessage(content=_REFUSAL)],
            )

        # Clean (PII/secrets already redacted in `sanitized`) → continue downstream
        # with the sanitized prompt so agents never see raw sensitive data. Carry the
        # per-scanner scores + a redaction flag so the trace UI can explain WHY it
        # passed (all blocking scanners below threshold), not just that it did.
        return patch(
            state,
            user_input=res["sanitized"],
            guard_block=None,
            guard_input={"scanned": True, "blocked": False, "findings": [],
                         "scores": res["scores"], "redacted": res["sanitized"] != text},
        )


class OutputGuard(Observable):
    """Scan the synthesized answer before it reaches the user."""

    _traced_methods: tuple[str, ...] = ("run",)
    _component_kind: str = "guard"
    _span_type: str = SpanType.CHAIN

    def run(self, state: AgentState) -> AgentState:
        """Scan ``final_output``; replace it with a refusal on a block, else emit the
        sanitized answer. No-op when there is no output to scan."""
        if get_settings().debug_break:
            breakpoint()  # opt-in: only fires when DEBUG_BREAK is set (see .vscode/launch.json)
        out = state.get("final_output") or ""
        if not out:
            return state

        guard = get_llm_guard()
        res = guard.scan_output(state.get("user_input") or "", out)

        if not res["valid"]:
            self.log("warning", "output_guard.blocked", findings=res["findings"])
            return patch(
                state,
                guard_block={"stage": "output", "findings": res["findings"], "scores": res["scores"]},
                guard_output={"scanned": True, "blocked": True, "findings": res["findings"],
                              "scores": res["scores"], "redacted": False},
                final_output=_REFUSAL,
                view=None,
                messages=[AIMessage(content=_REFUSAL)],
            )

        # Passed (PII redacted in the answer text) → emit the sanitized answer.
        return patch(
            state,
            final_output=res["sanitized"],
            guard_output={"scanned": True, "blocked": False, "findings": [],
                          "scores": res["scores"], "redacted": res["sanitized"] != out},
        )
