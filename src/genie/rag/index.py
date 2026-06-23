"""Dependency-free BM25 document index — the platform's RAG retrieval core.

Loads the framework's own markdown docs as a corpus, chunks them by section, and
serves top-k retrieval with a compact BM25 ranking. No vector DB, no embedding
API — sparse lexical retrieval keeps RAG self-contained and offline-testable
while still being real retrieval.

This module is the single home of the index logic. It backs both the standalone
RAG service ([services/rag/server.py]) and the in-process MCP ``search_docs`` tool
(which now re-exports from here via ``services/mcp/rag_index.py``).

The corpus root defaults to the repository root (``RAG_DOCS_DIR`` overrides it);
``*.md`` files at the root and anything under ``docs/`` are indexed.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from genie.platform.config import get_settings

# src/genie/rag/index.py -> repo root is four levels up (rag -> genie -> src -> root).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MAX_CHUNK_CHARS = 700
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")

# BM25 parameters.
_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> list[str]:
    """Lowercase the text and split into alphanumeric terms for BM25 matching."""
    return _TOKEN_RE.findall(text.lower())


def _docs_root() -> Path:
    """Resolve the corpus root: ``RAG_DOCS_DIR`` if set, else the repo root."""
    override = get_settings().rag_docs_dir
    return Path(override).resolve() if override else _REPO_ROOT


def _corpus_files(root: Path) -> list[Path]:
    """Collect indexable markdown: ``*.md`` at the root plus everything under ``docs/``."""
    files = sorted(root.glob("*.md"))
    docs_dir = root / "docs"
    if docs_dir.is_dir():
        files += sorted(docs_dir.rglob("*.md"))
    return files


def _chunk_file(path: Path, root: Path) -> list[dict]:
    """Split one markdown file into section-aware chunks of <= _MAX_CHUNK_CHARS."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.name

    heading = ""
    chunks: list[dict] = []
    buf: list[str] = []

    def flush() -> None:
        """Emit the buffered lines as one heading-tagged chunk and reset the buffer."""
        body = "\n".join(buf).strip()
        if body:
            source = f"{rel}#{heading}" if heading else rel
            chunks.append({"source": source, "text": body})
        buf.clear()

    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        m = _HEADING_RE.match(block.splitlines()[0])
        if m:
            flush()
            heading = m.group(1).strip()
        # Start a new chunk when adding this block would overflow the budget.
        current_len = sum(len(b) for b in buf)
        if buf and current_len + len(block) > _MAX_CHUNK_CHARS:
            flush()
        buf.append(block)
    flush()
    return chunks


class DocIndex:
    """In-memory BM25 index over a list of ``{"source", "text"}`` chunks."""

    def __init__(self, chunks: list[dict]) -> None:
        """Tokenize the chunks and precompute BM25 term-frequency, length, and IDF statistics."""
        self.chunks = chunks
        self._tokens = [_tokenize(c["text"]) for c in chunks]
        self._tf = [Counter(toks) for toks in self._tokens]
        self._len = [len(toks) for toks in self._tokens]
        self.N = len(chunks)
        self._avgdl = (sum(self._len) / self.N) if self.N else 0.0
        df: Counter = Counter()
        for toks in self._tokens:
            for term in set(toks):
                df[term] += 1
        self._df = df

    def _idf(self, term: str) -> float:
        """BM25 inverse document frequency for a term; 0 when it appears in no chunk."""
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Return the top-k chunks ranked by BM25 score for the query (empty if no match)."""
        q_terms = _tokenize(query)
        if not q_terms or self.N == 0:
            return []
        scored: list[tuple[float, int]] = []
        for i in range(self.N):
            tf, dl = self._tf[i], self._len[i] or 1
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + _K1 * (1 - _B + _B * dl / (self._avgdl or 1))
                score += self._idf(term) * (f * (_K1 + 1)) / denom
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda s: s[0], reverse=True)
        out = []
        for score, i in scored[:k]:
            out.append({
                "source": self.chunks[i]["source"],
                "text": self.chunks[i]["text"],
                "score": round(score, 4),
            })
        return out


_index: DocIndex | None = None


def get_index() -> DocIndex:
    """Lazily build (and cache) the corpus index."""
    global _index
    if _index is None:
        root = _docs_root()
        chunks: list[dict] = []
        for path in _corpus_files(root):
            chunks.extend(_chunk_file(path, root))
        _index = DocIndex(chunks)
    return _index
