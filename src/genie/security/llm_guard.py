"""Input/output content guard backed by the local ``llm-guard`` library.

Enabled by default, controlled by a single master switch
(``settings.llm_guard_enabled`` / ``LLM_GUARD_ENABLED``): when off, the graph
omits both guard nodes and the models below are never loaded (see
``genie.application.graph.build_graph``) — the pipeline then runs UNPROTECTED.
When on, the library import and model construction happen eagerly in
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

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from genie.observability import get_logger
from genie.platform.config import get_settings

_log = get_logger(__name__)

# Where scripts/quantize_guard_models.py writes the INT8 ONNX classifiers.
_QUANTIZED_DIR = Path(__file__).resolve().parent.parent / "models" / "guard_onnx"

# Scanners whose failure BLOCKS the request. Everything else only sanitizes.
_BLOCKING_INPUT = {"PromptInjection", "Toxicity", "BanTopics", "Regex"}
_BLOCKING_OUTPUT = {"Toxicity", "BanTopics"}

# Scanners that REWRITE the text (PII / secret redaction) rather than just
# classify it. They must run as an ordered chain (each sees the prior one's
# redactions); the read-only classifiers can run concurrently. In the scanner
# lists below the sanitizers always come LAST, so the classifiers already see
# the raw text today — running them in parallel on that same raw text is
# therefore semantically identical to the sequential scan_prompt/scan_output.
_SANITIZING_SCANNERS = {"Anonymize", "Secrets", "Sensitive"}

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
    """Banned topics from LLM_GUARD_BAN_TOPICS (comma-separated) or the defaults."""
    raw = get_settings().llm_guard_ban_topics
    if raw:
        return [t.strip() for t in raw.split(",") if t.strip()]
    return list(_DEFAULT_BAN_TOPICS)


def _injection_patterns() -> list[str]:
    """Code/script-injection regexes from LLM_GUARD_INJECTION_PATTERNS or the defaults."""
    # Regexes can contain commas, so split the override on newlines, not commas.
    raw = get_settings().llm_guard_injection_patterns
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
        self._fail_open = get_settings().llm_guard_fail_open
        # Optional ONNX-runtime backend for the ML scanners. OFF by default: on this
        # CPU benchmarked ~3x SLOWER per scan and ~10x slower at startup than PyTorch,
        # because llm-guard's default ONNX models are non-quantized (the CPU win comes
        # from INT8 quantization, which these aren't). Left as an opt-in toggle for
        # setups with quantized ONNX models. Requires `pip install llm-guard[onnxruntime]`.
        self._use_onnx = get_settings().llm_guard_use_onnx
        # Run the read-only classifiers concurrently instead of summing their
        # latencies sequentially. On (default) ON the wall-time of a scan drops
        # to ~the slowest single scanner. Each scanner owns a distinct model, so
        # concurrent forward passes are safe. Set LLM_GUARD_PARALLEL=0 to fall
        # back to llm-guard's sequential scan_prompt/scan_output.
        self._parallel = get_settings().llm_guard_parallel
        # EXPERIMENTAL, OFF by default. Use the locally-built INT8 ONNX classifiers
        # (scripts/quantize_guard_models.py) for the blocking scanners. Verdict on this
        # CPU (security/bench_guard.py): dynamic INT8 was ~3x SLOWER than PyTorch *and*
        # regressed PromptInjection — quantization pushed the DeBERTa injection score
        # under its block threshold, so jailbreaks passed. DO NOT enable without
        # re-running bench_guard and confirming both latency AND detection parity. Kept
        # as a gated toggle so better-quantized models can be re-evaluated. Falls back
        # per-model (logged) if a quantized file is missing, so it never aborts startup.
        self._quantized = get_settings().llm_guard_onnx_quantized
        qmodels = self._build_quantized_models() if self._quantized else {}
        # PII / secret redaction scanners. ON by default. These are SANITIZING (they
        # never block) and the heaviest non-blocking cost — Anonymize/Sensitive each
        # run a NER model. Set LLM_GUARD_PII=0 to drop them when the app handles no
        # personal data, removing a NER forward pass from BOTH guards. The BLOCKING
        # scanners (injection/toxicity/banned-topics/regex) are unaffected.
        self._pii = get_settings().llm_guard_pii
        topics = _ban_topics()
        injection_patterns = _injection_patterns()
        self._vault = Vault()

        # Eager construction forces model download/load now (startup), not on the
        # first user request — so load failures are caught at boot.
        self._input_scanners = [
            PromptInjection(**self._classifier_kwargs(qmodels, "prompt_injection")),
            InputToxicity(**self._classifier_kwargs(qmodels, "toxicity")),
            BanTopics(topics=topics, **self._classifier_kwargs(qmodels, "ban_topics")),
            # Deterministic code/script-injection backstop (blocks on any match).
            Regex(patterns=injection_patterns, is_blocked=True, match_type="search", redact=False),
        ]
        if self._pii:
            self._input_scanners += [Anonymize(self._vault, use_onnx=self._use_onnx), Secrets()]
        self._output_scanners = [
            OutputToxicity(**self._classifier_kwargs(qmodels, "toxicity")),
            OutputBanTopics(topics=topics, **self._classifier_kwargs(qmodels, "ban_topics")),
        ]
        if self._pii:
            self._output_scanners.append(Sensitive(use_onnx=self._use_onnx))
        _log.info(
            "llm_guard.scanners_ready",
            extra={"attrs": {
                "input": [type(s).__name__ for s in self._input_scanners],
                "output": [type(s).__name__ for s in self._output_scanners],
                "ban_topics": topics,
                "injection_patterns": len(injection_patterns),
                "quantized": sorted(qmodels.keys()),
                "fail_open": self._fail_open,
                "use_onnx": self._use_onnx,
            }},
        )

    # ------------------------------------------------------------------
    def _build_quantized_models(self) -> dict[str, Any]:
        """Map scanner key -> a ``Model`` pointing at the local INT8 ONNX file.

        Starts from each scanner's *default* Model and only swaps the ONNX path
        (via ``dataclasses.replace``), so the model's tokenizer/pipeline kwargs —
        which the scanner relies on for thresholds, truncation, label handling —
        are preserved exactly. A missing quantized file is skipped (and logged) so
        the scanner falls back to its default backend instead of failing closed.
        """
        import dataclasses

        from llm_guard.input_scanners import ban_topics, prompt_injection, toxicity

        defaults = {
            "prompt_injection": prompt_injection.V2_MODEL,
            "toxicity": toxicity.DEFAULT_MODEL,
            "ban_topics": ban_topics.MODEL_ROBERTA_BASE_C_V2,
        }
        out: dict[str, Any] = {}
        for key, model in defaults.items():
            onnx_dir = _QUANTIZED_DIR / key
            if not (onnx_dir / "model_quantized.onnx").exists():
                _log.warning(
                    "llm_guard.quantized_missing",
                    extra={"attrs": {"key": key, "dir": str(onnx_dir)}},
                )
                continue
            out[key] = dataclasses.replace(
                model,
                onnx_path=str(onnx_dir),
                onnx_subfolder="",
                onnx_filename="model_quantized.onnx",
            )
        return out

    def _classifier_kwargs(self, qmodels: dict[str, Any], key: str) -> dict[str, Any]:
        """Constructor kwargs for one blocking classifier: the quantized ONNX model
        when available, else the default backend (PyTorch, or FP32 ONNX if
        LLM_GUARD_USE_ONNX=1)."""
        model = qmodels.get(key)
        if model is not None:
            return {"model": model, "use_onnx": True}
        return {"use_onnx": self._use_onnx}

    # ------------------------------------------------------------------
    def warm(self) -> None:
        """Run one benign input+output scan so the models' kernels are warm.

        Construction (in ``__init__``) loads the weights, but the FIRST inference
        still pays a cold-kernel penalty (~340ms here). Calling this at startup —
        like the router's intent classifier — keeps that off the first real
        request. Best-effort: warming must never abort startup.
        """
        try:
            self.scan_input("hello")
            self.scan_output("hello", "hi there")
        except Exception as e:  # warming is best-effort
            _log.warning("llm_guard.warm_failed", extra={"attrs": {"error": str(e)}})

    # ------------------------------------------------------------------
    def _scan_parallel(
        self,
        scanners: list,
        scan_one: Callable[[Any, str], tuple[str, bool, float]],
        base_text: str,
    ) -> tuple[str, dict[str, bool], dict[str, float]]:
        """Concurrent equivalent of llm-guard's scan_prompt/scan_output.

        ``scan_one(scanner, text)`` runs one scanner and returns its
        ``(sanitized, is_valid, risk)`` triple. Read-only classifiers run
        concurrently on ``base_text``; the sanitizing scanners run as one ordered
        chain (preserving redaction composition) concurrently with them. Returns
        the same ``(sanitized, valid_map, scores)`` shape as scan_prompt, so the
        callers are unchanged. Any scanner error propagates so the caller can
        fail closed.
        """
        classifiers = [s for s in scanners if type(s).__name__ not in _SANITIZING_SCANNERS]
        sanitizers = [s for s in scanners if type(s).__name__ in _SANITIZING_SCANNERS]
        valid: dict[str, bool] = {}
        scores: dict[str, float] = {}

        def run_classifier(scanner: Any) -> tuple[str, bool, float]:
            _, is_valid, risk = scan_one(scanner, base_text)
            return type(scanner).__name__, bool(is_valid), float(risk)

        def run_sanitizer_chain() -> tuple[str, list[tuple[str, bool, float]]]:
            sanitized = base_text
            out: list[tuple[str, bool, float]] = []
            for scanner in sanitizers:
                sanitized, is_valid, risk = scan_one(scanner, sanitized)
                out.append((type(scanner).__name__, bool(is_valid), float(risk)))
            return sanitized, out

        sanitized = base_text
        workers = max(1, len(classifiers) + (1 if sanitizers else 0))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            cls_futures = [ex.submit(run_classifier, s) for s in classifiers]
            san_future = ex.submit(run_sanitizer_chain) if sanitizers else None
            for fut in cls_futures:
                name, is_valid, risk = fut.result()  # re-raises scanner errors
                valid[name], scores[name] = is_valid, risk
            if san_future is not None:
                sanitized, chain = san_future.result()
                for name, is_valid, risk in chain:
                    valid[name], scores[name] = is_valid, risk
        return sanitized, valid, scores

    # ------------------------------------------------------------------
    def _fail_result(self, text: str, stage: str, error: str) -> dict[str, Any]:
        """Result for a scanner runtime error: block (fail-closed) unless fail-open."""
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
            if self._parallel:
                sanitized, valid, scores = self._scan_parallel(
                    self._input_scanners, lambda s, t: s.scan(t), text
                )
            else:
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
            if self._parallel:
                sanitized, valid, scores = self._scan_parallel(
                    self._output_scanners, lambda s, t: s.scan(prompt or "", t), output
                )
            else:
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
    """Return the process-wide LLMGuard singleton, constructing (loading models) once."""
    global _store
    if _store is None:
        _store = LLMGuard()
    return _store
