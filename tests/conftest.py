"""Shared fixtures: a fake embedder that can be paused and that detects overlapping calls."""

import hashlib
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.app.vectorstore.faiss_store import FaissVectorStore

# Generous: only reached when a test is about to fail, so a hang becomes an assertion.
GATE_TIMEOUT_S = 10.0


def _vector(text: str, size: int) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [digest[i % len(digest)] / 255.0 - 0.5 for i in range(size)]


class ControllableEmbeddings(Embeddings):
    """Deterministic vectors, plus hooks to hold `embed_documents` mid-call.

    `entered` is set when a documents call starts; it then waits on `gate` (open by default).
    `max_in_flight` records the highest number of simultaneous calls, which should be 1 if the
    store serialises its embedder.
    """

    def __init__(self, size: int = 16) -> None:
        self._size = size
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.gate.set()
        self._counter_lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0

    def _enter(self) -> None:
        with self._counter_lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)

    def _exit(self) -> None:
        with self._counter_lock:
            self._in_flight -= 1

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._enter()
        try:
            self.entered.set()
            self.gate.wait(GATE_TIMEOUT_S)
            return [_vector(t, self._size) for t in texts]
        finally:
            self._exit()

    def embed_query(self, text: str) -> list[float]:
        self._enter()
        try:
            return _vector(text, self._size)
        finally:
            self._exit()


@pytest.fixture
def embeddings() -> Iterator[ControllableEmbeddings]:
    emb = ControllableEmbeddings()
    yield emb
    emb.gate.set()  # never leave a worker thread parked after a failed test


MODEL_NAME = "test-embedder"


def make_chunks(doc_id: str, n: int) -> tuple[list[Document], list[str]]:
    """`n` chunks of document `doc_id` with ids in the ingestion format `<doc_id>:<i>`."""
    docs = [
        Document(
            page_content=f"{doc_id} chunk {i}",
            metadata={"doc_id": doc_id, "chunk_index": i},
        )
        for i in range(n)
    ]
    return docs, [f"{doc_id}:{i}" for i in range(n)]


@pytest.fixture
def store(tmp_path: Path, embeddings: ControllableEmbeddings) -> FaissVectorStore:
    return FaissVectorStore(tmp_path / "index", embeddings, MODEL_NAME)
