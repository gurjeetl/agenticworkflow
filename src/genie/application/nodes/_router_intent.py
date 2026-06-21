"""Local, embedding-based multi-intent detector for the Router.

The Router's job is cheap triage *before* the expensive Planner. A clearly
multi-intent prompt always falls through to the Planner, so detecting it without
an LLM call saves ~1.5-2.5s. The regex heuristic in ``router_agent`` catches only
explicit additive phrasing ("...outages. ALSO the weather"); this classifier adds
the implicit cases ("weather in Tokyo and the top outages") by *counting how many
distinct registered agents the prompt activates*.

How: embed the prompt and each agent's capability text (description + tags) with a
small local sentence-transformers model (all-MiniLM-L6-v2, CPU, ~10-30ms), then
count agents whose cosine similarity to the prompt exceeds a threshold. ``>=
min_agents`` activated ⇒ multi-intent.

Fully local — no network, no big LLM. **Fails open**: any import/load/encode error
returns count 0, so the Router behaves exactly as before (its LLM call still runs).
Agent embeddings are cached and only recomputed when the active agent set changes.
"""
from __future__ import annotations

from genie.platform.config import get_settings
from genie.observability import get_logger

_log = get_logger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Cosine above which a prompt is considered to "touch" an agent. Calibrated against
# the bundled agents (see router/test_intent.py); override per-deployment.
_DEFAULT_THRESHOLD = 0.30
_DEFAULT_MIN_AGENTS = 2  # >= this many activated agents ⇒ multi-intent


class IntentClassifier:
    """Counts distinct agents a prompt activates, to flag multi-intent prompts."""

    def __init__(self) -> None:
        self._model_name = get_settings().router_intent_model
        self._threshold = get_settings().router_intent_threshold
        self._min_agents = get_settings().router_intent_min_agents
        self._enabled = get_settings().router_intent_classifier
        self._model = None
        self._agent_vecs: dict[str, object] = {}  # agent_id -> normalized embedding
        self._agent_sig: tuple | None = None  # signature of the cached agent set

    @property
    def enabled(self) -> bool:
        """True when the local intent classifier is active (model loadable)."""
        return self._enabled

    # ------------------------------------------------------------------
    @staticmethod
    def _agent_text(meta) -> str:
        """Capability text used to represent an agent in embedding space."""
        tags = " ".join(getattr(meta, "capability_tags", []) or [])
        return f"{getattr(meta, 'description', '') or ''} {tags}".strip()

    def _ensure_model(self):
        """Lazily load and cache the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            _log.info("router.intent_model_loaded", extra={"attrs": {"model": self._model_name}})
        return self._model

    def _ensure_agent_vecs(self, metas) -> None:
        """(Re)embed agent capability texts, cached until the active set changes."""
        sig = tuple(sorted((m.agent_id, getattr(m, "version", "")) for m in metas))
        if sig == self._agent_sig and self._agent_vecs:
            return
        texts = [self._agent_text(m) for m in metas]
        vecs = self._model.encode(texts, normalize_embeddings=True)
        self._agent_vecs = {m.agent_id: vecs[i] for i, m in enumerate(metas)}
        self._agent_sig = sig

    # ------------------------------------------------------------------
    def count_agents(self, text: str, metas) -> int:
        """Number of distinct agents the prompt activates (cosine >= threshold).

        Returns 0 on any failure (fail-open) so the caller keeps prior behavior.
        """
        if not self._enabled or not text or not metas:
            return 0
        try:
            model = self._ensure_model()
            self._ensure_agent_vecs(metas)
            q = model.encode([text], normalize_embeddings=True)[0]
            sims = {aid: float(q @ vec) for aid, vec in self._agent_vecs.items()}
        except Exception as e:  # missing dep, load failure, encode error → fail open
            _log.warning("router.intent_classifier_failed", extra={"attrs": {"error": str(e)}})
            return 0
        activated = [aid for aid, s in sims.items() if s >= self._threshold]
        _log.info(
            "router.intent_scored",
            extra={"attrs": {
                "activated": activated,
                "sims": {k: round(v, 3) for k, v in sims.items()},
                "threshold": self._threshold,
            }},
        )
        return len(activated)

    def is_multi_intent(self, text: str, metas) -> bool:
        """True when the prompt activates >= min_agents distinct agents."""
        return self.count_agents(text, metas) >= self._min_agents

    def warm(self) -> None:
        """Eagerly load the model (call at startup to avoid a cold first request)."""
        if self._enabled:
            try:
                self._ensure_model()
            except Exception as e:
                _log.warning("router.intent_warm_failed", extra={"attrs": {"error": str(e)}})


_classifier: IntentClassifier | None = None


def get_intent_classifier() -> IntentClassifier:
    """Return the process-wide classifier, constructing it once (shares the agent-vec cache)."""
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier
