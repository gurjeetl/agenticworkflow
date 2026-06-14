"""Mandatory input/output content guard backed by the local ``llm-guard`` library.

Unlike the optional stores (Redis/Milvus), this is **not** optional and has no
enable/disable flag. The library import and model construction happen eagerly in
``LLMGuard.__init__`` so that a missing dependency or un-loadable model surfaces
as a hard startup failure (fail-closed) rather than silently running the pipeline
unprotected.

Two classes of scanners:
  * BLOCKING  — a failure short-circuits the pipeline to a safe refusal
                (prompt injection, toxicity/harmful, banned topics).
  * SANITIZING — never blocks; redacts in place (PII via Anonymize/Sensitive,
                 credentials via Secrets). Their redaction is applied to the
                 returned ``sanitized`` text regardless of the block decision.

Singleton via ``get_llm_guard()`` mirrors the ``get_redis_store`` accessor in
memory/redis_store.py, minus the optional/no-op branch.
"""
from __future__ import annotations

import os
from typing import Any

from observability import get_logger

_log = get_logger(__name__)

# Scanners whose failure BLOCKS the request. Everything else only sanitizes.
_BLOCKING_INPUT = {"PromptInjection", "Toxicity", "BanTopics", "Regex"}
_BLOCKING_OUTPUT = {"Toxicity", "BanTopics"}

_DEFAULT_BAN_TOPICS = ["violence", "self-harm", "hate speech", "illegal weapons"]

# Deterministic defense-in-depth against application-layer code/script injection
# (XSS, SQLi, shell, template/Log4Shell, path traversal). This is NOT a substitute
# for fixing the layer where input is used — output-encoding when rendering, para-
# meterized queries, and never passing user input to a shell/eval. It is a cheap,
# 0ms backstop that blocks known payload patterns the ML scanners catch only by
# accident. The `{{...}}` and `$(` patterns carry the highest false-positive risk;
# override the whole list via LLM_GUARD_INJECTION_PATTERNS (newline-delimited).
_DEFAULT_INJECTION_PATTERNS = [
    # XSS / HTML injection
    r"(?i)<\s*script\b",
    r"(?i)\bon(error|load|click|mouseover)\s*=",
    r"(?i)javascript\s*:",
    r"(?i)<\s*iframe\b",
    # SQL injection
    r"(?i)\bunion\s+select\b",
    r"(?i);\s*drop\s+table\b",
    r"(?i)\bor\s+1\s*=\s*1\b",
    # Shell / command injection
    r"(?i);\s*rm\s+-rf\b",
    r"(?i)\|\s*(sh|bash)\b",
    r"\$\(",
    # Template / Log4Shell / SSTI
    r"(?i)\$\{\s*jndi:",
    r"\{\{.*\}\}",
    # Path traversal (require >=2 segments to cut false positives)
    r"(?:\.\.[\\/]){2,}",
]


def _ban_topics() -> list[str]:
    raw = os.getenv("LLM_GUARD_BAN_TOPICS")
    if raw:
        return [t.strip() for t in raw.split(",") if t.strip()]
    return list(_DEFAULT_BAN_TOPICS)


def _injection_patterns() -> list[str]:
    # Regexes can contain commas, so split the override on newlines, not commas.
    raw = os.getenv("LLM_GUARD_INJECTION_PATTERNS")
    if raw:
        return [p.strip() for p in raw.splitlines() if p.strip()]
    return list(_DEFAULT_INJECTION_PATTERNS)


class LLMGuard:
    """Eagerly-loaded local content guard. Constructed once at startup."""

    def __init__(self) -> None:
        # Plain imports: an ImportError propagates and aborts startup by design.
        from llm_guard.input_scanners import (
            Anonymize,
            BanTopics,
            PromptInjection,
            Regex,
            Secrets,
            Toxicity as InputToxicity,
        )
        from llm_guard.output_scanners import (
            BanTopics as OutputBanTopics,
            Sensitive,
            Toxicity as OutputToxicity,
        )
        from llm_guard.vault import Vault

        # Fail-closed by default: a scanner RUNTIME error blocks the request.
        # Set LLM_GUARD_FAIL_OPEN=1 to allow-through on scanner errors instead.
        self._fail_open = os.getenv("LLM_GUARD_FAIL_OPEN", "0") == "1"
        # Optional ONNX-runtime backend for the ML scanners. OFF by default: on this
        # CPU benchmarked ~3x SLOWER per scan and ~10x slower at startup than PyTorch,
        # because llm-guard's default ONNX models are non-quantized (the CPU win comes
        # from INT8 quantization, which these aren't). Left as an opt-in toggle for
        # setups with quantized ONNX models. Requires `pip install llm-guard[onnxruntime]`.
        self._use_onnx = os.getenv("LLM_GUARD_USE_ONNX", "0") == "1"
        topics = _ban_topics()
        injection_patterns = _injection_patterns()
        self._vault = Vault()

        # Eager construction forces model download/load now (startup), not on the
        # first user request — so load failures are caught at boot.
        self._input_scanners = [
            PromptInjection(use_onnx=self._use_onnx),
            InputToxicity(use_onnx=self._use_onnx),
            BanTopics(topics=topics, use_onnx=self._use_onnx),
            # Deterministic code/script-injection backstop (blocks on any match).
            Regex(patterns=injection_patterns, is_blocked=True, match_type="search", redact=False),
            Anonymize(self._vault, use_onnx=self._use_onnx),
            Secrets(),
        ]
        self._output_scanners = [
            OutputToxicity(use_onnx=self._use_onnx),
            OutputBanTopics(topics=topics, use_onnx=self._use_onnx),
            Sensitive(use_onnx=self._use_onnx),
        ]
        _log.info(
            "llm_guard.scanners_ready",
            extra={"attrs": {
                "input": [type(s).__name__ for s in self._input_scanners],
                "output": [type(s).__name__ for s in self._output_scanners],
                "ban_topics": topics,
                "injection_patterns": len(injection_patterns),
                "fail_open": self._fail_open,
                "use_onnx": self._use_onnx,
            }},
        )

    # ------------------------------------------------------------------
    def _fail_result(self, text: str, stage: str, error: str) -> dict[str, Any]:
        _log.warning("llm_guard.scan_error", extra={"attrs": {"stage": stage, "error": error}})
        if self._fail_open:
            return {"valid": True, "sanitized": text, "findings": [], "scores": {}}
        return {"valid": False, "sanitized": text, "findings": ["scan_error"], "scores": {}}

    def scan_input(self, text: str) -> dict[str, Any]:
        """Scan a user prompt. Returns {valid, sanitized, findings, scores}.

        ``valid`` is False only when a BLOCKING scanner fails; ``sanitized``
        always carries PII/secret redactions from the sanitizing scanners.
        """
        if not text:
            return {"valid": True, "sanitized": text, "findings": [], "scores": {}}
        try:
            from llm_guard import scan_prompt
            sanitized, valid, scores = scan_prompt(self._input_scanners, text)
        except Exception as e:  # fail-closed unless LLM_GUARD_FAIL_OPEN=1
            return self._fail_result(text, "input", str(e))
        findings = [name for name, ok in valid.items() if not ok and name in _BLOCKING_INPUT]
        if findings:
            _log.warning("llm_guard.input_blocked", extra={"attrs": {"findings": findings, "scores": scores}})
        return {"valid": not findings, "sanitized": sanitized, "findings": findings, "scores": scores}

    def scan_output(self, prompt: str, output: str) -> dict[str, Any]:
        """Scan the final answer against the prompt. Same contract as scan_input."""
        if not output:
            return {"valid": True, "sanitized": output, "findings": [], "scores": {}}
        try:
            from llm_guard import scan_output as _scan_output
            sanitized, valid, scores = _scan_output(self._output_scanners, prompt or "", output)
        except Exception as e:  # fail-closed unless LLM_GUARD_FAIL_OPEN=1
            return self._fail_result(output, "output", str(e))
        findings = [name for name, ok in valid.items() if not ok and name in _BLOCKING_OUTPUT]
        if findings:
            _log.warning("llm_guard.output_blocked", extra={"attrs": {"findings": findings, "scores": scores}})
        return {"valid": not findings, "sanitized": sanitized, "findings": findings, "scores": scores}


_store: LLMGuard | None = None


def get_llm_guard() -> LLMGuard:
    global _store
    if _store is None:
        _store = LLMGuard()
    return _store
