import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytesseract
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError

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
        recovered = 0
        for doc, text in zip(sparse, texts, strict=True):
            # A short typed page (title block, cover) must not be replaced by worse OCR output.
            if len(text.strip()) > len(doc.page_content.strip()):
                doc.page_content = text
                doc.metadata["ocr"] = True
                recovered += 1
        if recovered == 0:
            _log_loud(
                f"OCR recovered NO text from {len(sparse)} page(s) of {path.name}. "
                "Check the errors above (missing binaries or unreadable scan)."
            )

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
        except (PDFInfoNotInstalledError, pytesseract.TesseractNotFoundError) as exc:
            # Raced past the up-front check (e.g. PATH changed) -- still a setup problem.
            _log_loud(f"OCR failed for {path.name} page {page}: system dependency missing ({exc})")
            return ""
        except Exception as exc:  # rendering/tesseract raise many unrelated types
            logger.error("OCR failed for %s page %d: %s", path.name, page, exc, exc_info=True)
            return ""


def _ocr_available() -> bool:
    """Check the system binaries once per document, so a missing install is one clear error
    instead of one warning per page."""
    if shutil.which("pdftoppm") is None:
        _log_loud(
            "OCR SKIPPED: poppler-utils (pdftoppm) is not installed or not on PATH. "
            "Scanned PDFs will be rejected. Docker: rebuild the api image "
            "(`docker compose build --no-cache api`). Local: install poppler "
            "(apt install poppler-utils / brew install poppler / Windows poppler release) "
            "and add its bin/ to PATH."
        )
        return False
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        _log_loud(
            "OCR SKIPPED: the tesseract binary is not installed or not on PATH. "
            "Scanned PDFs will be rejected. Docker: rebuild the api image "
            "(`docker compose build --no-cache api`). Local: install Tesseract "
            "(apt install tesseract-ocr / brew install tesseract / UB-Mannheim Windows "
            "installer) and add it to PATH."
        )
        return False
    except Exception:
        # Present but broken (bad install, missing libs): don't mask it as "not installed".
        logger.exception("OCR check failed: tesseract is present but did not run")
        return False
    return True


def _log_loud(message: str) -> None:
    """Emit an error that stands out in a busy console; OCR misconfiguration otherwise
    surfaces to users only as a vague 'no extractable text' 422."""
    bar = "=" * 78
    logger.error("\n%s\n%s\n%s", bar, message, bar)
