from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PdfLoader:
    def load(self, path: Path) -> list[Document]:
        docs = PyPDFLoader(str(path)).load()
        for doc in docs:
            # PyPDFLoader pages are 0-indexed; humans cite from 1.
            doc.metadata = {"page": int(doc.metadata.get("page", 0)) + 1}
        return docs

    def validate_magic(self, head: bytes) -> bool:
        return head.lstrip().startswith(b"%PDF-")
