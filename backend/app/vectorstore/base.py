"""Vector store contract. Services depend on this, never on FAISS directly."""

from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document


@dataclass(frozen=True, slots=True)
class SearchHit:
    document: Document
    relevance: float  # 0..1, higher is better
    rerank_score: float | None = None  # cross-encoder probability (0..1); None if not reranked


def chunk_key(document: Document) -> str:
    """Stable identity of a chunk across retrievers; mirrors the ingestion id `<doc_id>:<n>`."""
    return f"{document.metadata.get('doc_id', '')}:{document.metadata.get('chunk_index', '')}"


class VectorStore(Protocol):
    def add(self, chunks: list[Document], ids: list[str]) -> None: ...

    def delete(self, ids: list[str]) -> None: ...

    def search(
        self, query: str, k: int, document_ids: frozenset[str] | None = None
    ) -> list[SearchHit]: ...

    @property
    def size(self) -> int: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]: ...
