"""Extension -> loader mapping. Adding a format = one loader class + one line here."""

from pathlib import Path

from backend.app.errors import UnsupportedFileTypeError
from backend.app.loaders.base import DocumentLoader
from backend.app.loaders.word import DocxLoader
from backend.app.loaders.pdf import PdfLoader
from backend.app.loaders.text import TextLoader

_LOADERS: dict[str, DocumentLoader] = {
    ".pdf": PdfLoader(),
    ".docx": DocxLoader(),
    ".txt": TextLoader(),
    ".md": TextLoader(),
}


def get_loader(path_or_name: str | Path) -> DocumentLoader:
    suffix = Path(path_or_name).suffix.lower()
    try:
        return _LOADERS[suffix]
    except KeyError:
        supported = ", ".join(sorted(_LOADERS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix or 'none'}'. Supported: {supported}."
        ) from None


def supported_extensions() -> frozenset[str]:
    return frozenset(_LOADERS)
