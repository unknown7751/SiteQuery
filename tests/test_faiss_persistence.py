"""Crash-safety of the on-disk index: generations, pointer swap, fsync order, validation."""

import json
import shutil
from pathlib import Path

import pytest
from langchain_community.vectorstores import FAISS

from backend.app import fsutil
from backend.app.vectorstore import EmbeddingMismatchError, IndexCorruptedError
from backend.app.vectorstore.faiss_store import FaissVectorStore
from tests.conftest import MODEL_NAME, ControllableEmbeddings, make_chunks


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def _pointer(directory: Path) -> dict[str, object]:
    loaded: dict[str, object] = json.loads((directory / "current").read_text())
    return loaded


def _open(directory: Path, embeddings: ControllableEmbeddings, model: str = MODEL_NAME) -> FaissVectorStore:
    return FaissVectorStore(directory, embeddings, model)


# ---- layout, metadata, pruning ----------------------------------------------------------


def test_persist_writes_generation_files_and_pointer_with_metadata(
    tmp_path: Path, embeddings: ControllableEmbeddings
) -> None:
    index = tmp_path / "index"
    _open(index, embeddings).add(*make_chunks("a", 3))

    assert _names(index) == ["current", "index.1.faiss", "index.1.pkl"]
    assert _pointer(index) == {
        "version": 1,
        "generation": 1,
        "embedding_model": MODEL_NAME,
        "dimension": 16,
        "vectors": 3,
    }


def test_each_write_is_a_new_generation_and_old_ones_are_pruned(
    tmp_path: Path, embeddings: ControllableEmbeddings
) -> None:
    index = tmp_path / "index"
    store = _open(index, embeddings)
    for doc in ("a", "b", "c"):
        store.add(*make_chunks(doc, 2))

    assert _pointer(index)["generation"] == 3
    # live + previous kept (a manual rollback point); generation 1 is gone
    assert _names(index) == ["current", "index.2.faiss", "index.2.pkl", "index.3.faiss", "index.3.pkl"]
    assert _open(index, embeddings).size == 6


def test_stray_files_from_a_crashed_write_are_ignored_then_pruned(
    tmp_path: Path, embeddings: ControllableEmbeddings
) -> None:
    index = tmp_path / "index"
    _open(index, embeddings).add(*make_chunks("a", 2))
    for name in ("index.9.faiss", "index.9.pkl", "current.tmp"):
        (index / name).write_bytes(b"torn write")

    store = _open(index, embeddings)  # pointer says generation 1; the debris is irrelevant
    assert store.size == 2
    store.add(*make_chunks("b", 1))
    assert not {"index.9.faiss", "index.9.pkl", "current.tmp"} & set(_names(index))


# ---- durability ordering ----------------------------------------------------------------


def test_files_and_directory_are_fsynced_before_the_pointer_flips(
    tmp_path: Path, embeddings: ControllableEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    target = "backend.app.vectorstore.faiss_store"

    def recording_fsync_file(path: Path) -> None:
        events.append(f"fsync {path.name}")
        fsutil.fsync_file(path)

    def recording_fsync_dir(path: Path) -> None:
        events.append("fsync dir")
        fsutil.fsync_dir(path)

    def recording_atomic_write(path: Path, data: bytes) -> None:
        events.append(f"flip {path.name}")
        fsutil.atomic_write_bytes(path, data)

    monkeypatch.setattr(f"{target}.fsync_file", recording_fsync_file)
    monkeypatch.setattr(f"{target}.fsync_dir", recording_fsync_dir)
    monkeypatch.setattr(f"{target}.atomic_write_bytes", recording_atomic_write)

    _open(tmp_path / "index", embeddings).add(*make_chunks("a", 2))

    assert events == ["fsync index.1.faiss", "fsync index.1.pkl", "fsync dir", "flip current"]


# ---- crashes: the previous generation must survive intact -------------------------------


def test_crash_before_pointer_flip_keeps_previous_generation(
    tmp_path: Path, embeddings: ControllableEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = tmp_path / "index"
    store = _open(index, embeddings)
    store.add(*make_chunks("a", 3))

    def disk_full(path: Path, data: bytes) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr("backend.app.vectorstore.faiss_store.atomic_write_bytes", disk_full)
    with pytest.raises(OSError, match="No space"):
        store.add(*make_chunks("b", 4))
    monkeypatch.undo()

    # memory rolled back to the last durable state, not left running ahead of disk
    assert store.size == 3
    assert store.chunk_ids() == frozenset({"a:0", "a:1", "a:2"})
    assert {h.document.metadata["doc_id"] for h in store.search("chunk", k=10)} == {"a"}
    # and a fresh process sees the same thing, with no torn faiss/pkl pair
    assert _open(index, embeddings).size == 3

    store.add(*make_chunks("c", 2))  # the store keeps working afterwards
    assert _open(index, embeddings).size == 5


def test_crash_midway_through_writing_the_new_pair_keeps_previous_generation(
    tmp_path: Path, embeddings: ControllableEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = tmp_path / "index"
    store = _open(index, embeddings)
    store.add(*make_chunks("a", 3))

    def killed_after_faiss_file(self: FAISS, folder_path: str, index_name: str = "index") -> None:
        (Path(folder_path) / f"{index_name}.faiss").write_bytes(b"half a write")
        raise OSError("process killed")

    monkeypatch.setattr(FAISS, "save_local", killed_after_faiss_file)
    with pytest.raises(OSError, match="killed"):
        store.add(*make_chunks("b", 4))
    monkeypatch.undo()

    assert (index / "index.2.faiss").exists()  # the torn file is really there...
    reopened = _open(index, embeddings)  # ...and harmless: nothing points at it
    assert reopened.size == 3
    assert reopened.search("a chunk 0", k=3)


def test_persist_failure_on_first_add_leaves_store_empty(
    store: FaissVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "_persist", boom)
    with pytest.raises(OSError):
        store.add(*make_chunks("a", 2))
    assert store.size == 0
    assert store.search("a chunk", k=3) == []


def test_persist_failure_on_delete_restores_the_vectors(
    store: FaissVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.add(*make_chunks("a", 3))

    def boom() -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "_persist", boom)
    with pytest.raises(OSError):
        store.delete(["a:0", "a:1"])
    assert store.size == 3
    assert store.chunk_ids() == frozenset({"a:0", "a:1", "a:2"})


# ---- tolerant delete --------------------------------------------------------------------


def test_delete_ignores_unknown_and_duplicate_ids_and_persists(
    tmp_path: Path, embeddings: ControllableEmbeddings
) -> None:
    index = tmp_path / "index"
    store = _open(index, embeddings)
    store.add(*make_chunks("a", 3))

    store.delete(["a:0", "a:0", "ghost:7"])

    assert store.chunk_ids() == frozenset({"a:1", "a:2"})
    assert _open(index, embeddings).chunk_ids() == frozenset({"a:1", "a:2"})


def test_delete_of_only_unknown_ids_writes_nothing(
    tmp_path: Path, embeddings: ControllableEmbeddings
) -> None:
    index = tmp_path / "index"
    store = _open(index, embeddings)
    store.add(*make_chunks("a", 2))

    store.delete(["ghost:0", "ghost:1"])
    store.delete([])

    assert _pointer(index)["generation"] == 1


# ---- loading: corruption ----------------------------------------------------------------


def test_truncated_faiss_file_is_reported_not_swallowed(
    tmp_path: Path, embeddings: ControllableEmbeddings
) -> None:
    index = tmp_path / "index"
    _open(index, embeddings).add(*make_chunks("a", 3))
    (index / "index.1.faiss").write_bytes(b"\x00" * 10)

    with pytest.raises(IndexCorruptedError, match=r"index\.1"):
        _open(index, embeddings)


def test_unreadable_pointer_is_reported(tmp_path: Path, embeddings: ControllableEmbeddings) -> None:
    index = tmp_path / "index"
    _open(index, embeddings).add(*make_chunks("a", 3))
    (index / "current").write_text("{not json")

    with pytest.raises(IndexCorruptedError, match="pointer"):
        _open(index, embeddings)


def test_pointer_to_missing_generation_is_reported(
    tmp_path: Path, embeddings: ControllableEmbeddings
) -> None:
    index = tmp_path / "index"
    _open(index, embeddings).add(*make_chunks("a", 3))
    (index / "index.1.faiss").unlink()

    with pytest.raises(IndexCorruptedError):
        _open(index, embeddings)


def test_mismatched_faiss_pkl_pair_is_rejected(tmp_path: Path, embeddings: ControllableEmbeddings) -> None:
    """The failure mode of the old two-file swap, now caught at load instead of at first query."""
    index = tmp_path / "index"
    store = _open(index, embeddings)
    store.add(*make_chunks("a", 3))
    store.add(*make_chunks("b", 3))  # generation 2 has 6 vectors
    shutil.copy(index / "index.1.pkl", index / "index.2.pkl")  # docstore maps only 3

    with pytest.raises(IndexCorruptedError, match="does not belong together"):
        _open(index, embeddings)


# ---- loading: embedding model validation ------------------------------------------------


def test_different_embedding_model_name_is_rejected(
    tmp_path: Path, embeddings: ControllableEmbeddings
) -> None:
    index = tmp_path / "index"
    _open(index, embeddings).add(*make_chunks("a", 2))

    with pytest.raises(EmbeddingMismatchError, match=f"{MODEL_NAME}.*other-model"):
        _open(index, embeddings, model="other-model")


def test_same_name_but_different_dimension_is_rejected(tmp_path: Path, embeddings: ControllableEmbeddings) -> None:
    index = tmp_path / "index"
    _open(index, embeddings).add(*make_chunks("a", 2))

    with pytest.raises(EmbeddingMismatchError, match="16-dimensional"):
        _open(index, ControllableEmbeddings(size=8))


def test_matching_model_reopens_cleanly(tmp_path: Path, embeddings: ControllableEmbeddings) -> None:
    index = tmp_path / "index"
    _open(index, embeddings).add(*make_chunks("a", 2))
    assert _open(index, embeddings).size == 2


# ---- legacy (pre-generation) indexes ----------------------------------------------------


def _write_legacy_index(directory: Path, embeddings: ControllableEmbeddings, n: int) -> None:
    chunks, ids = make_chunks("old", n)
    FAISS.from_documents(chunks, embeddings, ids=ids).save_local(str(directory), index_name="index")


def test_legacy_index_is_loaded_and_migrated_on_next_write(
    tmp_path: Path, embeddings: ControllableEmbeddings
) -> None:
    index = tmp_path / "index"
    _write_legacy_index(index, embeddings, 3)
    assert _names(index) == ["index.faiss", "index.pkl"]

    store = _open(index, embeddings)
    assert store.size == 3
    store.add(*make_chunks("new", 2))

    assert _names(index) == ["current", "index.1.faiss", "index.1.pkl"]
    assert _pointer(index)["embedding_model"] == MODEL_NAME
    assert _open(index, embeddings).size == 5


def test_legacy_index_dimension_is_still_checked(tmp_path: Path, embeddings: ControllableEmbeddings) -> None:
    index = tmp_path / "index"
    _write_legacy_index(index, embeddings, 3)

    with pytest.raises(EmbeddingMismatchError, match="16-dimensional"):
        _open(index, ControllableEmbeddings(size=8))
