"""Cross-encoder reranking: re-score the fused candidates with a model that reads query and chunk together.

Bi-encoder (FAISS) and BM25 scores are cheap but coarse. A cross-encoder attends over the
(query, chunk) pair jointly, which is far better at "does this passage actually answer the
question" — too slow for the whole corpus, ideal for the ~20 survivors of RRF.
"""

import logging
import math
import threading
from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder

from backend.app.vectorstore.base import SearchHit

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """FastEmbed ONNX cross-encoder (no torch, no API key); call via `asyncio.to_thread`."""

    def __init__(self, model_name: str, cache_dir: Path, min_score: float = 0.0) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._min_score = min_score
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir))
        # ONNX sessions are not guaranteed re-entrant across concurrent /query requests.
        self._lock = threading.Lock()

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        """Return up to `top_n` hits scoring >= `min_score`, best first, with `rerank_score` set.

        May return fewer than `top_n`: weak chunks cost LLM tokens and invite answers built on
        irrelevant text, so they are dropped rather than padded in. Never returns empty for
        non-empty input: if nothing clears `min_score` the single best chunk is kept (top-1
        fallback), because cross-encoders under-score casual phrasing and the LLM is better
        placed than a score cutoff to say "the documents don't contain this".

        The model emits raw logits; they are squashed with a sigmoid so the UI and API get a
        0..1 value. `relevance` (RRF rank agreement) is left untouched.
        """
        if not hits:
            return []
        texts = [h.document.page_content for h in hits]
        with self._lock:
            logits = list(self._model.rerank(query, texts))
        scored = [
            SearchHit(document=h.document, relevance=h.relevance, rerank_score=_sigmoid(logit))
            for h, logit in zip(hits, logits, strict=True)
        ]
        scored.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
        kept = [h for h in scored if (h.rerank_score or 0.0) >= self._min_score][:top_n]
        if not kept:
            kept = scored[:1]
            logger.info(
                "No candidate cleared rerank threshold %.2f; keeping top-1 (score %.3f)",
                self._min_score, kept[0].rerank_score or 0.0,
            )
        logger.debug(
            "Reranked %d candidates -> %d kept (threshold %.2f, top_n=%d)",
            len(hits), len(kept), self._min_score, top_n,
        )
        return kept


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))
