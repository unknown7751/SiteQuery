"""FAISS-backed vector store with crash-safe on-disk persistence. All FAISS code lives here."""

import json
import logging
import re
import shutil
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.app.fsutil import atomic_write_bytes, fsync_dir, fsync_file
from backend.app.vectorstore.base import SearchHit
from backend.app.vectorstore.errors import EmbeddingMismatchError, IndexCorruptedError

logger = logging.getLogger(__name__)

_EMBED_BATCH = 32
_DIM_PROBE = "dimension probe"

# On-disk layout, all inside the index directory:
#   index.<gen>.faiss / index.<gen>.pkl   one immutable file pair per generation
#   current                               JSON pointer: which generation is live + metadata
# Files are never overwritten in place; a write creates generation N+1 and then flips `current`
# with a single atomic rename, so a crash leaves either the old or the new generation, never a
# .faiss from one and a .pkl from the other.
_POINTER = "current"
_POINTER_VERSION = 1
_LEGACY_NAME = "index"  # pre-generation layout: index.faiss / index.pkl, no pointer
_LEGACY_FILES = ("index.faiss", "index.pkl")
_GENERATION_FILE = re.compile(r"^index\.(\d+)\.(?:faiss|pkl)$")
_KEEP_GENERATIONS = 2  # live + previous, so a bad write can be rolled back by hand


@dataclass(frozen=True, slots=True)
class _Pointer:
    generation: int
    embedding_model: str
    dimension: int
    vectors: int

    def dumps(self) -> bytes:
        return json.dumps({"version": _POINTER_VERSION, **asdict(self)}, indent=2).encode("utf-8")

    @classmethod
    def loads(cls, raw: bytes) -> Self:
        data = json.loads(raw)
        if data["version"] != _POINTER_VERSION:
            raise ValueError(f"unsupported pointer version {data['version']!r}")
        return cls(
            generation=int(data["generation"]),
            embedding_model=str(data["embedding_model"]),
            dimension=int(data["dimension"]),
            vectors=int(data["vectors"]),
        )


class FaissVectorStore:
    """Thread-safe wrapper; call its methods via `asyncio.to_thread`.

    Two locks, never held together:
    - `_lock` guards the FAISS index and its persistence. It is held only for index
      mutation / lookup, never while embedding, so a slow upload cannot stall queries.
    - `_embed_lock` serialises the embedder. ONNX sessions are re-entrant, but the
      tokenizer in front of them is not guaranteed to be, so calls are kept one at a time.
      Documents are embedded in small batches so a query can slot in between them.

    The index is created lazily on first `add` (FAISS needs a vector to learn the dimension).
    Every mutation is persisted as a new generation (see the layout note above); if the write
    fails, memory is reloaded from the last durable generation so memory never runs ahead of
    disk. Startup refuses to load an index that is corrupt or was built with a different
    embedding model than `embedding_model`.
    """

    def __init__(self, index_dir: Path, embeddings: Embeddings, embedding_model: str) -> None:
        self._dir = index_dir
        self._embeddings = embeddings
        self._model_name = embedding_model
        self._lock = threading.Lock()
        self._embed_lock = threading.Lock()
        self._generation = 0
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

    def chunk_ids(self) -> frozenset[str]:
        """Every chunk id in the index; lets startup reconcile it against the manifest."""
        with self._lock:
            if self._db is None:
                return frozenset()
            return frozenset(self._db.index_to_docstore_id.values())

    def add(self, chunks: list[Document], ids: list[str]) -> None:
        if len(chunks) != len(ids):
            raise ValueError("chunks and ids must have the same length")
        texts = [c.page_content for c in chunks]
        pairs = list(zip(texts, self._embed_documents(texts), strict=True))  # outside `_lock`
        metadatas = [c.metadata for c in chunks]
        with self._lock:
            try:
                if self._db is None:
                    self._db = FAISS.from_embeddings(pairs, self._embeddings, metadatas=metadatas, ids=ids)
                else:
                    self._db.add_embeddings(pairs, metadatas=metadatas, ids=ids)
                self._size = self._count()
                self._persist()
            except BaseException:
                self._restore_durable()
                raise

    def delete(self, ids: list[str]) -> None:
        """Remove `ids`; ids not in the index are ignored.

        LangChain's `FAISS.delete` raises if any id is unknown, which would make a
        half-ingested or already-removed document impossible to delete.
        """
        with self._lock:
            if self._db is None:
                return
            present = set(self._db.index_to_docstore_id.values())
            doomed = [i for i in dict.fromkeys(ids) if i in present]
            if not doomed:
                return
            try:
                self._db.delete(doomed)
                self._size = self._count()
                self._persist()
            except BaseException:
                self._restore_durable()
                raise

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
        db, pointer = self._open_durable()
        if db is None:
            logger.info("No existing FAISS index at %s; starting empty", self._dir)
            return None
        self._validate(db, pointer)
        logger.info("Loaded FAISS index generation %d with %d vectors", self._generation, db.index.ntotal)
        return db

    def _open_durable(self) -> tuple[FAISS | None, _Pointer | None]:
        """Read whatever `current` points at (or the legacy pair); sets `_generation`."""
        pointer = self._read_pointer()
        if pointer is not None:
            name, self._generation = f"index.{pointer.generation}", pointer.generation
        elif (self._dir / _LEGACY_FILES[0]).exists():
            logger.warning("Loading pre-generation index files; they are migrated on the next write")
            name, self._generation = _LEGACY_NAME, 0
        else:
            self._generation = 0
            return None, None
        try:
            # Safe: the pickle is written only by this service into a private volume.
            db = FAISS.load_local(
                str(self._dir), self._embeddings, index_name=name, allow_dangerous_deserialization=True
            )
        except Exception as exc:  # faiss, pickle and the filesystem raise unrelated types
            raise IndexCorruptedError(f"Could not load FAISS index '{name}' from {self._dir}: {exc}") from exc
        return db, pointer

    def _read_pointer(self) -> _Pointer | None:
        path = self._dir / _POINTER
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            return _Pointer.loads(raw)
        except (ValueError, KeyError, TypeError) as exc:
            raise IndexCorruptedError(f"Index pointer {path} is unreadable: {exc}") from exc

    def _validate(self, db: FAISS, pointer: _Pointer | None) -> None:
        ntotal, dim = int(db.index.ntotal), int(db.index.d)
        if ntotal != len(db.index_to_docstore_id):
            raise IndexCorruptedError(
                f"Index in {self._dir} holds {ntotal} vectors but its docstore maps "
                f"{len(db.index_to_docstore_id)}; the .faiss/.pkl pair does not belong together."
            )
        if pointer is not None:
            if (pointer.vectors, pointer.dimension) != (ntotal, dim):
                raise IndexCorruptedError(
                    f"Index generation {pointer.generation} does not match its pointer "
                    f"(pointer: {pointer.vectors} vectors/{pointer.dimension} dims, "
                    f"files: {ntotal}/{dim})."
                )
            if pointer.embedding_model != self._model_name:
                raise EmbeddingMismatchError(
                    f"The index in {self._dir} was built with embedding model "
                    f"'{pointer.embedding_model}', but EMBEDDING_MODEL is '{self._model_name}'. "
                    "Set EMBEDDING_MODEL back, or rebuild the index from the uploaded documents."
                )
        else:
            logger.warning("Index predates embedding metadata; only its dimension can be checked")
        configured_dim = len(self._embed_query(_DIM_PROBE))
        if configured_dim != dim:
            raise EmbeddingMismatchError(
                f"The index in {self._dir} has {dim}-dimensional vectors, but embedding model "
                f"'{self._model_name}' produces {configured_dim}. Set EMBEDDING_MODEL back, "
                "or rebuild the index from the uploaded documents."
            )

    def _persist(self) -> None:
        """Write generation N+1, make it durable, then flip `current` to it.

        Nothing the old generation depends on is touched, so any failure before the pointer
        rename leaves the previous generation fully intact (the stray N+1 files are pruned later).
        """
        if self._db is None:
            raise RuntimeError("nothing to persist")
        generation = self._generation + 1
        name = f"index.{generation}"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db.save_local(str(self._dir), index_name=name)
        for suffix in (".faiss", ".pkl"):
            fsync_file(self._dir / f"{name}{suffix}")
        fsync_dir(self._dir)  # the new file names must be durable before anything points at them
        pointer = _Pointer(
            generation=generation,
            embedding_model=self._model_name,
            dimension=int(self._db.index.d),
            vectors=int(self._db.index.ntotal),
        )
        atomic_write_bytes(self._dir / _POINTER, pointer.dumps())
        self._generation = generation
        self._prune(keep=set(range(generation - _KEEP_GENERATIONS + 1, generation + 1)))

    def _prune(self, keep: set[int]) -> None:
        """Delete superseded generations, failed-write leftovers and legacy files.

        Runs after the pointer flip, so a failure here must not fail the write.
        """
        try:
            stale = [
                p
                for p in self._dir.iterdir()
                if p.name in _LEGACY_FILES
                or p.name == f"{_POINTER}.tmp"
                or ((m := _GENERATION_FILE.match(p.name)) is not None and int(m.group(1)) not in keep)
            ]
            for path in stale:
                path.unlink(missing_ok=True)
            shutil.rmtree(self._dir / ".tmp", ignore_errors=True)  # staging dir of the old layout
        except OSError:
            logger.warning("Could not prune old index generations in %s", self._dir, exc_info=True)

    def _restore_durable(self) -> None:
        """After a failed write, drop in-memory changes by reloading the last durable state."""
        try:
            self._db, _ = self._open_durable()
        except Exception:
            logger.exception("Could not reload the last durable index; in-memory state may be stale")
        self._size = self._count()


def _to_relevance(squared_l2: float) -> float:
    """Map FAISS squared-L2 distance to 0..1.

    FastEmbed's BGE vectors are unit-normalised, so cosine = 1 - d²/2.
    """
    return max(0.0, min(1.0, 1.0 - squared_l2 / 2.0))
