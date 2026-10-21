"""Startup reconciliation of the vector index against the document manifest."""

import json
from pathlib import Path

import pytest

from backend.app.services.reconciliation import reconcile_index
from backend.app.services.registry import DocumentRegistry
from backend.app.vectorstore import HybridVectorStore
from backend.app.vectorstore.faiss_store import FaissVectorStore
from tests.conftest import MODEL_NAME, ControllableEmbeddings, make_chunks
from tests.test_registry import make_record


@pytest.fixture
def registry(tmp_path: Path) -> DocumentRegistry:
    return DocumentRegistry(tmp_path / "manifest" / "documents.json")


def _generation(directory: Path) -> int:
    return int(json.loads((directory / "current").read_text())["generation"])


def test_phantom_vectors_are_dropped_and_the_drop_is_durable(
    tmp_path: Path, embeddings: ControllableEmbeddings, registry: DocumentRegistry
) -> None:
    index = tmp_path / "index"
    dense = FaissVectorStore(index, embeddings, MODEL_NAME)
    dense.add(*make_chunks("a", 3))
    dense.add(*make_chunks("b", 2))  # crashed ingest: indexed, never recorded
    registry.add(make_record("a", chunks=3))

    result = reconcile_index(dense, registry)

    assert result.phantoms_dropped == 2
    assert dense.chunk_ids() == frozenset({"a:0", "a:1", "a:2"})
    assert FaissVectorStore(index, embeddings, MODEL_NAME).size == 3  # survives a restart


def test_keyword_index_built_after_reconciliation_has_no_phantoms(
    tmp_path: Path, embeddings: ControllableEmbeddings, registry: DocumentRegistry
) -> None:
    dense = FaissVectorStore(tmp_path / "index", embeddings, MODEL_NAME)
    dense.add(*make_chunks("a", 3))
    dense.add(*make_chunks("ghost", 2))
    registry.add(make_record("a", chunks=3))

    reconcile_index(dense, registry)
    hybrid = HybridVectorStore(dense)  # same order as the lifespan: reconcile, then build BM25

    assert hybrid.size == 3
    hits = hybrid.search("ghost chunk 0", k=10)  # a term only the phantom chunks contain
    assert hits and {h.document.metadata["doc_id"] for h in hits} == {"a"}


def test_consistent_index_is_left_alone(
    tmp_path: Path, embeddings: ControllableEmbeddings, registry: DocumentRegistry
) -> None:
    index = tmp_path / "index"
    dense = FaissVectorStore(index, embeddings, MODEL_NAME)
    dense.add(*make_chunks("a", 3))
    registry.add(make_record("a", chunks=3))
    before = _generation(index)

    result = reconcile_index(dense, registry)

    assert (result.phantoms_dropped, result.incomplete_documents, result.skipped_empty_manifest) == (0, [], False)
    assert _generation(index) == before  # no gratuitous rewrite


def test_empty_manifest_never_wipes_a_populated_index(
    tmp_path: Path, embeddings: ControllableEmbeddings, registry: DocumentRegistry
) -> None:
    """A lost or quarantined manifest looks exactly like this; dropping everything is unrecoverable."""
    dense = FaissVectorStore(tmp_path / "index", embeddings, MODEL_NAME)
    dense.add(*make_chunks("a", 3))

    result = reconcile_index(dense, registry)

    assert result.skipped_empty_manifest and result.phantoms_dropped == 0
    assert dense.size == 3


def test_manifest_documents_with_missing_chunks_are_reported_not_deleted(
    tmp_path: Path, embeddings: ControllableEmbeddings, registry: DocumentRegistry
) -> None:
    dense = FaissVectorStore(tmp_path / "index", embeddings, MODEL_NAME)
    dense.add(*make_chunks("a", 3))
    registry.add(make_record("a", chunks=3))
    registry.add(make_record("c", chunks=3))  # its vectors never made it to the index

    result = reconcile_index(dense, registry)

    assert result.incomplete_documents == [("c", 3)]
    assert registry.get("c") is not None  # reported only: removing a document is the user's call


def test_everything_empty_is_a_no_op(tmp_path: Path, embeddings: ControllableEmbeddings, registry: DocumentRegistry) -> None:
    dense = FaissVectorStore(tmp_path / "index", embeddings, MODEL_NAME)
    result = reconcile_index(dense, registry)
    assert result.phantoms_dropped == 0 and not result.skipped_empty_manifest
