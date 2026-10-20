from pathlib import Path

from langchain_core.documents import Document

_ENCODINGS = ("utf-8-sig", "utf-16", "cp1252")


class TextLoader:
    """Plain text / Markdown. Tries a few encodings common in exported project files."""

    def load(self, path: Path) -> list[Document]:
        raw = path.read_bytes()
        for encoding in _ENCODINGS:
            try:
                return [Document(page_content=raw.decode(encoding), metadata={})]
            except UnicodeError:
                continue
        raise ValueError("Unsupported text encoding")

    def validate_magic(self, head: bytes) -> bool:
        return b"\x00" not in head or head.startswith((b"\xff\xfe", b"\xfe\xff"))
