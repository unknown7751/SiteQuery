"""Hybrid retrieval: dense (FAISS) + sparse (BM25), fused with Reciprocal Rank Fusion.

Dense embeddings capture meaning but blur exact identifiers ("ASTM-C39", "RFI-1042");
BM25 nails those but misses paraphrase. Running both and fusing by rank gets each one's wins.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document

from backend.app.vectorstore.base import SearchHit
from backend.app.vectorstore.bm25 import Bm25Index
from backend.app.vectorstore.faiss_store import FaissVectorStore
from backend.app.vectorstore.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class HybridVectorStore:
    """Implements `VectorStore`; call its methods via `asyncio.to_thread`.

    FAISS is the source of truth. The BM25 index is derived from it at construction and
    kept in step on add/delete, so the two always cover the same chunks.
    """

    def __init__(self, dense: FaissVectorStore, candidates: int = 20, rrf_k: int = 60) -> None:
        self._dense = dense
        self._candidates = candidates
        self._rrf_k = rrf_k
        self._sparse = Bm25Index(dense.documents())
        logger.info("Hybrid store ready: %d dense / %d keyword chunks", dense.size, self._sparse.size)

    @property
    def size(self) -> int:
        return self._dense.size

    def add(self, chunks: list[Document], ids: list[str]) -> None:
        self._dense.add(chunks, ids)  # persists first; raises before BM25 is touched
        self._sparse.add(list(zip(ids, chunks, strict=True)))

    def delete(self, ids: list[str]) -> None:
        self._dense.delete(ids)
        self._sparse.delete(ids)

    def search(
        self, query: str, k: int, document_ids: frozenset[str] | None = None
    ) -> list[SearchHit]:
        # Each retriever over-fetches so fusion has overlap to reward; trimmed to k after.
        pool = max(k, self._candidates)
        # The embedding call (ONNX) and BM25 scoring are independent: overlap them.
        with ThreadPoolExecutor(max_workers=1) as executor:
            sparse_future = executor.submit(self._sparse.search, query, pool, document_ids)
            dense_hits = self._dense.search(query, pool, document_ids)
            sparse_hits = sparse_future.result()
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], k=self._rrf_k)
        logger.debug(
            "Hybrid search: %d dense, %d keyword -> %d fused", len(dense_hits), len(sparse_hits), len(fused)
        )
        return fused[:k]
