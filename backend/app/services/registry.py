"""Persistent manifest of ingested documents (JSON file next to the FAISS index)."""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from backend.app.errors import ManifestCorruptedError
from backend.app.fsutil import atomic_write_bytes
from backend.app.schemas import DocumentInfo

logger = logging.getLogger(__name__)


class DocumentRecord(BaseModel):
    id: str
    filename: str
    stored_path: str
    sha256: str
    size_bytes: int
    pages: int | None
    chunk_ids: list[str]
    uploaded_at: datetime

    def to_info(self) -> DocumentInfo:
        return DocumentInfo(
            id=self.id,
            filename=self.filename,
            size_bytes=self.size_bytes,
            pages=self.pages,
            chunks=len(self.chunk_ids),
            uploaded_at=self.uploaded_at,
        )


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocumentRegistry:
    """Thread-safe; synchronous (tiny JSON file), call via `asyncio.to_thread` from routes."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._records: dict[str, DocumentRecord] = self._read()

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.uploaded_at, reverse=True)

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._records.get(doc_id)

    def find_by_hash(self, sha256: str) -> DocumentRecord | None:
        with self._lock:
            return next((r for r in self._records.values() if r.sha256 == sha256), None)

    def add(self, record: DocumentRecord) -> None:
        with self._lock:
            self._records[record.id] = record
            self._write()

    def remove(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            record = self._records.pop(doc_id, None)
            if record is not None:
                self._write()
            return record

    def _read(self) -> dict[str, DocumentRecord]:
        """Load the manifest; a corrupt one is quarantined and raises instead of resetting.

        Silently starting empty would be destructive: the index still holds every chunk, and
        the next write would overwrite what is left of the manifest. I/O errors (permissions,
        a flaky disk) propagate unchanged; only unparseable *content* is quarantined.
        """
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        try:
            raw = json.loads(text)
            return {r["id"]: DocumentRecord.model_validate(r) for r in raw}
        except (ValueError, KeyError, TypeError) as exc:  # ValueError covers JSON/UTF-8/pydantic
            quarantined = self._quarantine()
            logger.error("Document manifest %s is corrupt: %s", self._path, exc)
            raise ManifestCorruptedError(self._path, quarantined, str(exc)) from exc

    def _quarantine(self) -> Path | None:
        """Move the corrupt manifest aside without clobbering an earlier quarantined copy."""
        target = self._path.with_name(f"{self._path.name}.corrupt")
        n = 0
        while target.exists():
            n += 1
            target = self._path.with_name(f"{self._path.name}.corrupt.{n}")
        try:
            os.replace(self._path, target)
        except OSError:
            logger.exception("Could not quarantine %s", self._path)
            return None
        return target

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [r.model_dump(mode="json") for r in self._records.values()]
        atomic_write_bytes(self._path, json.dumps(payload, indent=2).encode("utf-8"))
