"""FAISS-backed vector store with on-disk persistence. All FAISS code lives here."""

import logging
import os
import shutil
import threading
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.app.vectorstore.base import SearchHit

logger = logging.getLogger(__name__)

_INDEX_NAME = "index"
_EMBED_BATCH = 32


class FaissVectorStore:
    """Thread-safe wrapper; call its methods via `asyncio.to_thread`.

    Two locks, never held together:
    - `_lock` guards the FAISS index and its persistence. It is held only for index
      mutation / lookup, never while embedding, so a slow upload cannot stall queries.
    - `_embed_lock` serialises the embedder. ONNX sessions are re-entrant, but the
      tokenizer in front of them is not guaranteed to be, so calls are kept one at a time.
      Documents are embedded in small batches so a query can slot in between them.

    The index is created lazily on first `add` (FAISS needs a vector to learn the dimension).
    Persistence stages the new index files and swaps them in with atomic renames.
    """

    def __init__(self, index_dir: Path, embeddings: Embeddings) -> None:
        self._dir = index_dir
        self._embeddings = embeddings
        self._lock = threading.Lock()
        self._embed_lock = threading.Lock()
        self._db: FAISS | None = self._load()
        # Plain int, written under `_lock`, read without it: `size` is called from async
        # routes, where waiting on `_lock` would freeze the event loop for a whole upload.
        self._size = self._count()

    # ---- public API -------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Vector count. Lock-free (an int read is atomic), so safe on the event loop."""
        return self._size

    def documents(self) -> list[tuple[str, Document]]:
        """Snapshot of every (chunk id, chunk) — used to seed the keyword index at startup."""
        with self._lock:
            if self._db is None:
                return []
            found = ((i, self._db.docstore.search(i)) for i in self._db.index_to_docstore_id.values())
            return [(i, doc) for i, doc in found if isinstance(doc, Document)]

    def add(self, chunks: list[Document], ids: list[str]) -> None:
        if len(chunks) != len(ids):
            raise ValueError("chunks and ids must have the same length")
        texts = [c.page_content for c in chunks]
        pairs = list(zip(texts, self._embed_documents(texts), strict=True))  # outside `_lock`
        metadatas = [c.metadata for c in chunks]
        with self._lock:
            if self._db is None:
                self._db = FAISS.from_embeddings(pairs, self._embeddings, metadatas=metadatas, ids=ids)
            else:
                self._db.add_embeddings(pairs, metadatas=metadatas, ids=ids)
            self._size = self._count()
            self._persist()

    def delete(self, ids: list[str]) -> None:
        with self._lock:
            if self._db is None or not ids:
                return
            self._db.delete(ids)
            self._size = self._count()
            self._persist()

    def search(
        self, query: str, k: int, document_ids: frozenset[str] | None = None
    ) -> list[SearchHit]:
        if self._size == 0:
            return []
        vector = self._embed_query(query)  # outside `_lock`
        metadata_filter = (
            (lambda meta: meta.get("doc_id") in document_ids)
            if document_ids is not None
            else None
        )
        with self._lock:
            if self._db is None or self._db.index.ntotal == 0:
                return []
            # fetch_k over-fetches so the filter still leaves up to k hits.
            results = self._db.similarity_search_with_score_by_vector(
                vector, k=k, filter=metadata_filter, fetch_k=max(k * 8, 50)
            )
        return [SearchHit(document=doc, relevance=_to_relevance(dist)) for doc, dist in results]

    # ---- embedding --------------------------------------------------------------------

    def _embed_query(self, text: str) -> list[float]:
        with self._embed_lock:
            return self._embeddings.embed_query(text)

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            with self._embed_lock:  # released between batches so queries are not starved
                vectors.extend(self._embeddings.embed_documents(texts[start : start + _EMBED_BATCH]))
        return vectors

    def _count(self) -> int:
        return 0 if self._db is None else int(self._db.index.ntotal)

    # ---- persistence ------------------------------------------------------------------

    def _load(self) -> FAISS | None:
        if not (self._dir / f"{_INDEX_NAME}.faiss").exists():
            logger.info("No existing FAISS index at %s; starting empty", self._dir)
            return None
        # Safe: the pickle is written only by this service into a private volume.
        db = FAISS.load_local(
            str(self._dir),
            self._embeddings,
            index_name=_INDEX_NAME,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded FAISS index with %d vectors", db.index.ntotal)
        return db

    def _persist(self) -> None:
        assert self._db is not None
        self._dir.mkdir(parents=True, exist_ok=True)
        # Stage inside the index dir: it may be a bind mount, and os.replace cannot cross
        # filesystems. The dir itself can't be renamed, so swap file-by-file (each atomic).
        tmp = self._dir / ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        self._db.save_local(str(tmp), index_name=_INDEX_NAME)
        for item in tmp.iterdir():
            os.replace(item, self._dir / item.name)
        shutil.rmtree(tmp, ignore_errors=True)


def _to_relevance(squared_l2: float) -> float:
    """Map FAISS squared-L2 distance to 0..1.

    FastEmbed's BGE vectors are unit-normalised, so cosine = 1 - d²/2.
    """
    return max(0.0, min(1.0, 1.0 - squared_l2 / 2.0))
