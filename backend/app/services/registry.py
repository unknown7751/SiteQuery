"""Persistent manifest of ingested documents (JSON file next to the FAISS index)."""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

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
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return {r["id"]: DocumentRecord.model_validate(r) for r in raw}
        except (OSError, ValueError, KeyError):
            logger.exception("Could not read document manifest %s; starting empty", self._path)
            return {}

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        payload = [r.model_dump(mode="json") for r in self._records.values()]
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)
