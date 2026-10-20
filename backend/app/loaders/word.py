from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document


class DocxLoader:
    def load(self, path: Path) -> list[Document]:
        docs = Docx2txtLoader(str(path)).load()
        for doc in docs:
            doc.metadata = {}  # DOCX has no stable page concept
        return docs

    def validate_magic(self, head: bytes) -> bool:
        return head.startswith(b"PK\x03\x04")  # DOCX is a zip container
