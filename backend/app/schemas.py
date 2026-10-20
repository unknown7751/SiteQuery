"""Pydantic models for every API boundary."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    id: str
    filename: str
    size_bytes: int
    pages: int | None = None
    chunks: int
    uploaded_at: datetime


class UploadResponse(BaseModel):
    document: DocumentInfo
    message: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    top_k: int | None = Field(default=None, ge=1, le=12)
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    document_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these documents."
    )


class SourceChunk(BaseModel):
    index: int = Field(description="1-based citation number used as [n] in the answer.")
    document_id: str
    filename: str
    page: int | None = None
    snippet: str
    relevance: float = Field(ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    documents: int
    chunks: int
    llm_configured: bool
    model: str
