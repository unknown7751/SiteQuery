from pathlib import Path

from langchain_core.documents import Document

_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")


def decode_text(raw: bytes) -> str:
    """Decode exported project text files.

    UTF-16 is only attempted when a BOM says so: the codec accepts almost any even-length
    byte string, so trying it speculatively turns Windows-1252 text (e.g. "20°F") into
    mojibake. cp1252 is the last resort; bytes it cannot map raise `UnicodeDecodeError`,
    which ingestion reports as an unreadable file.
    """
    if raw.startswith(_UTF16_BOMS):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


class TextLoader:
    """Plain text / Markdown. Handles the encodings common in exported project files."""

    def load(self, path: Path) -> list[Document]:
        return [Document(page_content=decode_text(path.read_bytes()), metadata={})]

    def validate_magic(self, head: bytes) -> bool:
        return b"\x00" not in head or head.startswith(_UTF16_BOMS)
