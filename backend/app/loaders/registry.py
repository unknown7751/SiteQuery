"""Extension -> loader mapping. Adding a format = one loader class + one line here."""

from collections.abc import Callable
from pathlib import Path

from backend.app.config import Settings
from backend.app.errors import UnsupportedFileTypeError
from backend.app.loaders.base import DocumentLoader
from backend.app.loaders.word import DocxLoader
from backend.app.loaders.pdf import PdfLoader
from backend.app.loaders.text import TextLoader

# Factories, not instances: settings are injected when a loader is requested, so importing
# this module reads no configuration.
_FACTORIES: dict[str, Callable[[Settings], DocumentLoader]] = {
    ".pdf": PdfLoader,
    ".docx": lambda _: DocxLoader(),
    ".txt": lambda _: TextLoader(),
    ".md": lambda _: TextLoader(),
}


def get_loader(path_or_name: str | Path, settings: Settings) -> DocumentLoader:
    suffix = Path(path_or_name).suffix.lower()
    try:
        factory = _FACTORIES[suffix]
    except KeyError:
        supported = ", ".join(sorted(_FACTORIES))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix or 'none'}'. Supported: {supported}."
        ) from None
    return factory(settings)


def supported_extensions() -> frozenset[str]:
    return frozenset(_FACTORIES)
