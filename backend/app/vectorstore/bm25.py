"""In-memory BM25 keyword index.

Not persisted: it is rebuilt from the FAISS docstore at startup, so the two can never drift
apart across restarts. `rank_bm25` cannot add documents incrementally, so every mutation
re-fits the model — fine at project scale (thousands of chunks, rare uploads).
"""

import heapq
import logging
import re
import threading
from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from backend.app.vectorstore.base import SearchHit

logger = logging.getLogger(__name__)

# An identifier such as "ASTM-C39", "RFI-1042" or "03.30.00" is one token...
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")
_PART = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens, emitting each compound identifier whole AND as its parts.

    So "ASTM-C39" matches the written forms "ASTM C39" and "ASTM-C39" alike, and
    "RFI #1042" still hits "RFI-1042" via the shared "1042" part.
    """
    tokens: list[str] = []
    for match in _TOKEN.findall(text.lower()):
        parts = _PART.findall(match)
        if len(parts) > 1:
            tokens.append(match)
        tokens.extend(parts)
    return tokens


@dataclass(frozen=True, slots=True)
class _Entry:
    document: Document
    tokens: list[str]
    vocab: frozenset[str]


class Bm25Index:
    """Thread-safe keyword index keyed by chunk id (same ids as the vector store)."""

    def __init__(self, documents: list[tuple[str, Document]] | None = None) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, _Entry] = {}
        self._ids: list[str] = []
        self._model: BM25Okapi | None = None
        if documents:
            self.add(documents)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def add(self, documents: list[tuple[str, Document]]) -> None:
        # Tokenise outside the lock; it is the expensive part.
        new: dict[str, _Entry] = {}
        for chunk_id, doc in documents:
            tokens = tokenize(doc.page_content)
            new[chunk_id] = _Entry(doc, tokens, frozenset(tokens))
        with self._lock:
            self._entries.update(new)
            self._refit()

    def delete(self, ids: list[str]) -> None:
        with self._lock:
            for chunk_id in ids:
                self._entries.pop(chunk_id, None)
            self._refit()

    def search(
        self, query: str, k: int, document_ids: frozenset[str] | None = None
    ) -> list[SearchHit]:
        """Top-`k` chunks sharing at least one term with `query`, best first.

        `relevance` is the raw BM25 score (unbounded); callers should rely on rank only.
        Chunks with no term overlap are excluded: BM25Okapi can score tiny corpora at or
        below zero, so overlap, not score sign, decides whether a chunk is a match.
        """
        query_terms = list(dict.fromkeys(tokenize(query)))
        with self._lock:
            if self._model is None or not query_terms:
                return []
            scores = self._model.get_scores(query_terms)
            wanted = set(query_terms)
            candidates: list[tuple[float, _Entry]] = []
            for i, chunk_id in enumerate(self._ids):
                entry = self._entries[chunk_id]
                if not wanted & entry.vocab:
                    continue
                if document_ids is not None and entry.document.metadata.get("doc_id") not in document_ids:
                    continue
                candidates.append((float(scores[i]), entry))
        best = heapq.nlargest(k, candidates, key=lambda pair: pair[0])
        return [SearchHit(document=entry.document, relevance=score) for score, entry in best]

    def _refit(self) -> None:
        self._ids = list(self._entries)
        corpus = [self._entries[i].tokens for i in self._ids]
        self._model = BM25Okapi(corpus) if any(corpus) else None
        logger.debug("BM25 index refit over %d chunks", len(self._ids))
