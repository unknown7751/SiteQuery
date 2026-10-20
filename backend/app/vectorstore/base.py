"""Vector store contract. Services depend on this, never on FAISS directly."""

from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document


@dataclass(frozen=True, slots=True)
class SearchHit:
    document: Document
    relevance: float  # 0..1, higher is better


class VectorStore(Protocol):
    def add(self, chunks: list[Document], ids: list[str]) -> None: ...

    def delete(self, ids: list[str]) -> None: ...

    def search(
        self, query: str, k: int, document_ids: frozenset[str] | None = None
    ) -> list[SearchHit]: ...

    @property
    def size(self) -> int: ...
