"""Upload -> parse -> chunk -> embed -> index pipeline."""

import asyncio
import logging
from pathlib import Path

from fastapi import UploadFile
from langchain_core.documents import Document

from backend.app.config import Settings
from backend.app.errors import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    InvalidFileError,
    NoExtractableTextError,
)
from backend.app.loaders import get_loader
from backend.app.schemas import DocumentInfo
from backend.app.services.chunking import build_splitter
from backend.app.services.registry import DocumentRecord, DocumentRegistry, utcnow
from backend.app.services.storage import StoredFile, save_upload
from backend.app.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self, settings: Settings, store: VectorStore, registry: DocumentRegistry
    ) -> None:
        self._settings = settings
        self._store = store
        self._registry = registry
        self._splitter = build_splitter(settings)

    async def ingest(self, upload: UploadFile) -> DocumentInfo:
        stored = await save_upload(upload, self._settings)
        try:
            existing = await asyncio.to_thread(self._registry.find_by_hash, stored.sha256)
            if existing is not None:
                raise DuplicateDocumentError(
                    f"'{existing.filename}' has already been uploaded.", existing.id
                )
            chunks, pages = await asyncio.to_thread(self._parse_and_split, stored)
            ids = [f"{stored.doc_id}:{i}" for i in range(len(chunks))]
            await asyncio.to_thread(self._store.add, chunks, ids)
            record = DocumentRecord(
                id=stored.doc_id,
                filename=stored.original_name,
                stored_path=str(stored.path),
                sha256=stored.sha256,
                size_bytes=stored.size_bytes,
                pages=pages,
                chunk_ids=ids,
                uploaded_at=utcnow(),
            )
            try:
                await asyncio.to_thread(self._registry.add, record)
            except BaseException:
                await asyncio.to_thread(self._store.delete, ids)  # keep index == manifest
                raise
        except BaseException:
            stored.path.unlink(missing_ok=True)
            raise
        logger.info("Ingested %s (%d chunks)", stored.original_name, len(ids))
        return record.to_info()

    async def delete(self, doc_id: str) -> None:
        record = await asyncio.to_thread(self._registry.get, doc_id)
        if record is None:
            raise DocumentNotFoundError(f"Document '{doc_id}' not found.")
        await asyncio.to_thread(self._store.delete, record.chunk_ids)
        await asyncio.to_thread(self._registry.remove, doc_id)
        Path(record.stored_path).unlink(missing_ok=True)
        logger.info("Deleted document %s (%s)", record.filename, doc_id)

    def _parse_and_split(self, stored: StoredFile) -> tuple[list[Document], int | None]:
        loader = get_loader(stored.original_name, self._settings)
        try:
            pages = loader.load(stored.path)
        except Exception as exc:  # parser libraries raise many unrelated types
            logger.warning("Failed to parse %s: %s", stored.original_name, exc)
            raise InvalidFileError(
                f"Could not read '{stored.original_name}'. The file may be corrupt or encrypted."
            ) from exc

        pages = [p for p in pages if p.page_content.strip()]
        if not pages:
            raise NoExtractableTextError(
                "No extractable text found. If this is a scan, OCR may be disabled or "
                "unavailable on the server, or the scan quality is too low to read."
            )

        chunks = self._splitter.split_documents(pages)
        page_numbers = [p.metadata["page"] for p in pages if "page" in p.metadata]
        for i, chunk in enumerate(chunks):
            chunk.metadata = {
                **chunk.metadata,
                "doc_id": stored.doc_id,
                "source": stored.original_name,
                "chunk_index": i,
            }
        return chunks, (max(page_numbers) if page_numbers else None)
