"""Loader contract. Loaders are synchronous; callers run them in a worker thread."""

from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document


class DocumentLoader(Protocol):
    def load(self, path: Path) -> list[Document]:
        """Return one Document per page/section.

        Metadata contract: `page` is 1-indexed and omitted when the format has no pages.
        """
        ...

    def validate_magic(self, head: bytes) -> bool:
        """Cheap content sniff on the first bytes, so `.pdf` files that aren't PDFs are rejected."""
        ...
