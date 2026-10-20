import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytesseract
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pdf2image import convert_from_path

from backend.app.config import Settings

logger = logging.getLogger(__name__)


class PdfLoader:
    """Extracts embedded text, falling back to OCR page-by-page.

    Per-page (not per-file) so mixed PDFs, e.g. a typed cover letter followed by scanned
    drawings, only pay for OCR where it is needed.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def load(self, path: Path) -> list[Document]:
        docs = PyPDFLoader(str(path)).load()
        for doc in docs:
            # PyPDFLoader pages are 0-indexed; humans cite from 1.
            doc.metadata = {"page": int(doc.metadata.get("page", 0)) + 1}
        if self._settings.ocr_enabled:
            self._ocr_sparse_pages(path, docs)
        return docs

    def validate_magic(self, head: bytes) -> bool:
        return head.lstrip().startswith(b"%PDF-")

    # ---- OCR --------------------------------------------------------------------------

    def _ocr_sparse_pages(self, path: Path, docs: list[Document]) -> None:
        cfg = self._settings
        sparse = [d for d in docs if len(d.page_content.strip()) < cfg.ocr_min_chars]
        if not sparse:
            return
        if not _ocr_available():
            return
        if len(sparse) > cfg.ocr_max_pages:
            logger.warning(
                "%s has %d pages needing OCR; only the first %d will be processed",
                path.name,
                len(sparse),
                cfg.ocr_max_pages,
            )
            sparse = sparse[: cfg.ocr_max_pages]

        logger.info("Running OCR on %d page(s) of %s", len(sparse), path.name)
        with ThreadPoolExecutor(max_workers=max(1, cfg.ocr_workers)) as pool:
            texts = list(pool.map(lambda d: self._ocr_page(path, d.metadata["page"]), sparse))
        for doc, text in zip(sparse, texts, strict=True):
            # A short typed page (title block, cover) must not be replaced by worse OCR output.
            if len(text.strip()) > len(doc.page_content.strip()):
                doc.page_content = text
                doc.metadata["ocr"] = True

    def _ocr_page(self, path: Path, page: int) -> str:
        """Render one page and OCR it. Failures are logged and yield '' so one bad page
        doesn't sink the whole upload."""
        cfg = self._settings
        try:
            images = convert_from_path(
                str(path), dpi=cfg.ocr_dpi, first_page=page, last_page=page
            )
            return "\n".join(
                pytesseract.image_to_string(
                    image, lang=cfg.ocr_language, timeout=cfg.ocr_page_timeout_s
                )
                for image in images
            )
        except Exception as exc:  # rendering/tesseract raise many unrelated types
            logger.warning("OCR failed for %s page %d: %s", path.name, page, exc)
            return ""


def _ocr_available() -> bool:
    """Check the system binaries once per document, so a missing install is one clear error
    instead of one warning per page."""
    if shutil.which("pdftoppm") is None:
        logger.error("OCR skipped: poppler-utils (pdftoppm) is not installed")
        return False
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        logger.error("OCR skipped: the tesseract-ocr binary is not installed")
        return False
    return True
