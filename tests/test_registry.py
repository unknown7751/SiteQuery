"""Manifest durability and corruption handling."""

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.app import fsutil
from backend.app.errors import ManifestCorruptedError
from backend.app.services.registry import DocumentRecord, DocumentRegistry


def make_record(doc_id: str, chunks: int = 2) -> DocumentRecord:
    return DocumentRecord(
        id=doc_id,
        filename=f"{doc_id}.txt",
        stored_path=f"/data/uploads/{doc_id}_{doc_id}.txt",
        sha256=f"hash-{doc_id}",
        size_bytes=100,
        pages=None,
        chunk_ids=[f"{doc_id}:{i}" for i in range(chunks)],
        uploaded_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    return tmp_path / "index" / "documents.json"


def test_missing_manifest_means_empty_library(manifest: Path) -> None:
    assert DocumentRegistry(manifest).list() == []


def test_records_survive_reload_and_leave_no_temp_file(manifest: Path) -> None:
    registry = DocumentRegistry(manifest)
    registry.add(make_record("a"))
    registry.add(make_record("b"))
    registry.remove("a")

    reloaded = DocumentRegistry(manifest)
    assert [r.id for r in reloaded.list()] == ["b"]
    assert sorted(p.name for p in manifest.parent.iterdir()) == ["documents.json"]


def test_manifest_is_written_through_the_durable_helper(
    manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsynced: list[int] = []
    real_fsync = os.fsync

    def record_fsync(fd: int) -> None:
        fsynced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", record_fsync)
    dirs: list[Path] = []

    def record_dir(path: Path) -> None:
        dirs.append(path)

    monkeypatch.setattr(fsutil, "fsync_dir", record_dir)

    DocumentRegistry(manifest).add(make_record("a"))

    assert fsynced, "manifest bytes were not fsynced before the rename"
    assert dirs == [manifest.parent], "rename was not made durable with a directory fsync"


# ---- corruption -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "why"),
    [
        ("{not json", "invalid JSON"),
        ("", "zero-length file left by a power loss after rename"),
        ('[{"id": "a"}]', "valid JSON, record fields missing"),
        ('{"a": 1}', "valid JSON, wrong shape"),
        ("42", "valid JSON, not a list"),
    ],
)
def test_corrupt_manifest_is_quarantined_and_raises(manifest: Path, content: str, why: str) -> None:
    manifest.parent.mkdir(parents=True)
    manifest.write_text(content)

    with pytest.raises(ManifestCorruptedError) as excinfo:
        DocumentRegistry(manifest)

    quarantined = manifest.with_name("documents.json.corrupt")
    assert excinfo.value.quarantined_to == quarantined
    assert quarantined.read_text() == content, why  # evidence preserved byte for byte
    assert not manifest.exists()  # nothing left that a later write could silently replace
    assert str(manifest) in str(excinfo.value) and str(quarantined) in str(excinfo.value)


def test_quarantine_never_overwrites_an_earlier_corrupt_copy(manifest: Path) -> None:
    manifest.parent.mkdir(parents=True)
    first = manifest.with_name("documents.json.corrupt")
    first.write_text("first incident")
    manifest.write_text("{second incident")

    with pytest.raises(ManifestCorruptedError) as excinfo:
        DocumentRegistry(manifest)

    assert first.read_text() == "first incident"
    assert excinfo.value.quarantined_to == manifest.with_name("documents.json.corrupt.1")
    assert excinfo.value.quarantined_to.read_text() == "{second incident"


def test_unreadable_manifest_is_not_mistaken_for_corrupt(
    manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An I/O error says nothing about the content, so nothing may be moved or reset."""
    manifest.parent.mkdir(parents=True)
    manifest.write_text("[]")

    def denied(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(PermissionError):
        DocumentRegistry(manifest)
    monkeypatch.undo()
    assert manifest.exists()


# ---- atomic write primitive -------------------------------------------------------------


def test_atomic_write_replaces_content_and_cleans_up(tmp_path: Path) -> None:
    target = tmp_path / "file.json"
    fsutil.atomic_write_bytes(target, b"one")
    fsutil.atomic_write_bytes(target, b"two")
    assert target.read_bytes() == b"two"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["file.json"]


def test_atomic_write_failure_leaves_original_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "file.json"
    fsutil.atomic_write_bytes(target, b"original")

    def fail(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError("rename failed")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        fsutil.atomic_write_bytes(target, b"new")
    monkeypatch.undo()
    assert target.read_bytes() == b"original"
