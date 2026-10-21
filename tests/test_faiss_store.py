import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from langchain_core.documents import Document

from backend.app.vectorstore.faiss_store import FaissVectorStore
from tests.conftest import GATE_TIMEOUT_S, ControllableEmbeddings

# Reading `size` must return near-instantly; anything near this bound means it blocked.
PROMPT_S = 1.0


def _chunks(doc_id: str, n: int) -> tuple[list[Document], list[str]]:
    docs = [
        Document(
            page_content=f"{doc_id} chunk {i}",
            metadata={"doc_id": doc_id, "chunk_index": i},
        )
        for i in range(n)
    ]
    return docs, [f"{doc_id}:{i}" for i in range(n)]


def _run_in_thread(fn: Callable[[], object]) -> threading.Thread:
    thread = threading.Thread(target=fn, daemon=True)
    thread.start()
    return thread


def _read_size_promptly(store: FaissVectorStore) -> int:
    """Read `store.size` from a helper thread and fail the test if it does not return."""
    result: list[int] = []
    reader = _run_in_thread(lambda: result.append(store.size))
    reader.join(PROMPT_S)
    assert not reader.is_alive(), "store.size blocked; it would freeze the event loop"
    return result[0]


# ---- behaviour preserved by the lock refactor ------------------------------------------


def test_add_search_roundtrip_and_size(store: FaissVectorStore) -> None:
    assert store.size == 0
    assert store.search("anything", k=3) == []

    chunks, ids = _chunks("a", 3)
    store.add(chunks, ids)

    assert store.size == 3
    hits = store.search("a chunk 1", k=3)
    assert {h.document.metadata["doc_id"] for h in hits} == {"a"}
    assert len(hits) == 3


def test_document_filter_and_delete_update_size(store: FaissVectorStore) -> None:
    for doc_id in ("a", "b"):
        store.add(*_chunks(doc_id, 3))
    assert store.size == 6

    only_b = store.search("chunk", k=6, document_ids=frozenset({"b"}))
    assert {h.document.metadata["doc_id"] for h in only_b} == {"b"}

    store.delete(["a:0", "a:1", "a:2"])
    assert store.size == 3


def test_index_survives_reload(tmp_path: Path, embeddings: ControllableEmbeddings) -> None:
    index_dir = tmp_path / "index"
    FaissVectorStore(index_dir, embeddings).add(*_chunks("a", 4))
    reloaded = FaissVectorStore(index_dir, embeddings)
    assert reloaded.size == 4


def test_large_add_is_embedded_in_batches(store: FaissVectorStore, embeddings: ControllableEmbeddings) -> None:
    calls: list[int] = []
    original = embeddings.embed_documents

    def spy(texts: list[str]) -> list[list[float]]:
        calls.append(len(texts))
        return original(texts)

    embeddings.embed_documents = spy  # type: ignore[method-assign]
    store.add(*_chunks("a", 70))
    assert len(calls) > 1 and sum(calls) == 70


# ---- C1: size must stay readable while an upload is in flight --------------------------


def test_size_readable_while_add_is_embedding(
    store: FaissVectorStore, embeddings: ControllableEmbeddings
) -> None:
    store.add(*_chunks("seed", 2))
    embeddings.gate.clear()
    embeddings.entered.clear()

    adder = _run_in_thread(lambda: store.add(*_chunks("slow", 5)))
    assert embeddings.entered.wait(GATE_TIMEOUT_S), "add() never started embedding"
    try:
        assert _read_size_promptly(store) == 2  # unchanged until the add commits
    finally:
        embeddings.gate.set()
    adder.join(GATE_TIMEOUT_S)
    assert store.size == 7


def test_size_readable_while_add_is_persisting(
    store: FaissVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_persist` runs under the index lock; the old locked `size` stalled right here."""
    in_persist = threading.Event()
    release = threading.Event()

    def slow_persist() -> None:
        in_persist.set()
        release.wait(GATE_TIMEOUT_S)

    monkeypatch.setattr(store, "_persist", slow_persist)
    adder = _run_in_thread(lambda: store.add(*_chunks("a", 4)))
    assert in_persist.wait(GATE_TIMEOUT_S), "add() never reached persist"
    try:
        assert _read_size_promptly(store) == 4  # in-memory index already holds the vectors
    finally:
        release.set()
    adder.join(GATE_TIMEOUT_S)


def test_embedder_calls_never_overlap(store: FaissVectorStore, embeddings: ControllableEmbeddings) -> None:
    store.add(*_chunks("seed", 3))
    errors: list[BaseException] = []

    def guarded(fn: Callable[[], object]) -> Callable[[], None]:
        def run() -> None:
            try:
                fn()
            except BaseException as exc:  # surfaced in the main thread below
                errors.append(exc)

        return run

    def add_job(doc_id: str) -> Callable[[], None]:
        return guarded(lambda: store.add(*_chunks(doc_id, 40)))

    jobs = [add_job(f"d{i}") for i in range(3)]
    jobs += [guarded(lambda: store.search("seed chunk", k=3)) for _ in range(6)]
    threads = [_run_in_thread(job) for job in jobs]
    for t in threads:
        t.join(GATE_TIMEOUT_S * 3)

    assert not errors
    assert not any(t.is_alive() for t in threads)
    assert embeddings.max_in_flight == 1
    assert store.size == 3 + 3 * 40


# ---- C2 (not fixed yet): FAISS.delete rejects unknown ids -------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="C2: langchain FAISS.delete raises on unknown ids, so a half-ingested or already "
    "removed document can never be deleted. Remove this marker when delete() tolerates them.",
)
def test_delete_tolerates_missing_ids(store: FaissVectorStore) -> None:
    store.add(*_chunks("a", 2))
    store.delete(["a:0", "ghost:0"])  # desired: unknown ids ignored, known ones removed
    assert store.size == 1
