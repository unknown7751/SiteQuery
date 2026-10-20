"""Typed HTTP client for the FastAPI backend. The UI never imports backend code."""

import os
from dataclasses import dataclass
from typing import Any

import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")


class ApiError(Exception):
    """Raised with a user-presentable message."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class DocumentItem:
    id: str
    filename: str
    size_bytes: int
    pages: int | None
    chunks: int


@dataclass(frozen=True, slots=True)
class Source:
    index: int
    filename: str
    page: int | None
    snippet: str
    relevance: float
    rerank_score: float | None
    ocr: bool


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    sources: list[Source]


@dataclass(frozen=True, slots=True)
class Health:
    documents: int
    chunks: int
    llm_configured: bool
    model: str


def _detail(resp: requests.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(str(d.get("msg", d)) for d in detail)
    return str(detail) if detail else f"Request failed ({resp.status_code})."


def _request(method: str, path: str, *, timeout: float, **kwargs: Any) -> requests.Response:
    try:
        resp = requests.request(method, f"{API_BASE_URL}{path}", timeout=timeout, **kwargs)
    except requests.ConnectionError as exc:
        raise ApiError("Cannot reach the API service. Is the backend running?") from exc
    except requests.Timeout as exc:
        raise ApiError("The request timed out. Please try again.") from exc
    if not resp.ok:
        raise ApiError(_detail(resp), resp.status_code)
    return resp


def get_health() -> Health:
    data = _request("GET", "/health", timeout=5).json()
    return Health(data["documents"], data["chunks"], data["llm_configured"], data["model"])


def list_documents() -> list[DocumentItem]:
    docs = _request("GET", "/documents", timeout=15).json()["documents"]
    return [
        DocumentItem(d["id"], d["filename"], d["size_bytes"], d["pages"], d["chunks"])
        for d in docs
    ]


def upload_document(filename: str, content: bytes, mime: str | None) -> str:
    resp = _request(
        "POST",
        "/upload_document",
        timeout=300,  # first upload may wait on embedding a large PDF
        files={"file": (filename, content, mime or "application/octet-stream")},
    )
    return str(resp.json()["message"])


def delete_document(doc_id: str) -> None:
    _request("DELETE", f"/documents/{doc_id}", timeout=30)


def ask(
    question: str,
    history: list[dict[str, str]],
    document_ids: list[str] | None,
    top_k: int,
) -> Answer:
    payload: dict[str, Any] = {
        "question": question,
        "history": history,
        "top_k": top_k,
        "document_ids": document_ids or None,
    }
    data = _request("POST", "/query", timeout=300, json=payload).json()
    sources = [
        Source(
            s["index"],
            s["filename"],
            s["page"],
            s["snippet"],
            s["relevance"],
            s.get("rerank_score"),
            bool(s.get("ocr", False)),
        )
        for s in data["sources"]
    ]
    return Answer(data["answer"], sources)
