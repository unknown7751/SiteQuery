"""Application settings, read once from environment variables."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


ReasoningEffort = Literal["low", "medium", "high"]
_EFFORTS: tuple[ReasoningEffort, ...] = ("low", "medium", "high")


def _env_effort(name: str, default: ReasoningEffort) -> ReasoningEffort:
    raw = os.environ.get(name, default).strip().lower()
    for effort in _EFFORTS:
        if raw == effort:
            return effort
    raise ValueError(f"{name} must be one of {', '.join(_EFFORTS)}; got '{raw}'")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if not raw else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    max_upload_mb: int
    allowed_extensions: frozenset[str]
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    default_top_k: int
    retrieval_candidates: int
    rrf_k: int
    rerank_enabled: bool
    rerank_model: str
    rerank_candidates: int
    rerank_threshold: float
    ocr_enabled: bool
    ocr_min_chars: int
    ocr_dpi: int
    ocr_language: str
    ocr_max_pages: int
    ocr_workers: int
    ocr_page_timeout_s: int
    groq_api_key: str | None
    groq_model: str
    groq_reasoning_effort: ReasoningEffort
    groq_max_tokens: int
    groq_timeout_s: int
    max_history_turns: int
    cors_origins: tuple[str, ...]

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def model_cache_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings(
        data_dir=Path(os.environ.get("DATA_DIR", str(_REPO_ROOT / "data"))).resolve(),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 50),
        allowed_extensions=frozenset({".pdf", ".docx", ".txt", ".md"}),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
        default_top_k=_env_int("DEFAULT_TOP_K", 5),
        retrieval_candidates=_env_int("RETRIEVAL_CANDIDATES", 20),
        rrf_k=_env_int("RRF_K", 60),
        rerank_enabled=_env_bool("RERANK_ENABLED", True),
        rerank_model=os.environ.get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
        rerank_candidates=_env_int("RERANK_CANDIDATES", 20),
        rerank_threshold=_env_float("RERANK_THRESHOLD", 0.10),
        ocr_enabled=_env_bool("OCR_ENABLED", True),
        ocr_min_chars=_env_int("OCR_MIN_CHARS", 50),
        ocr_dpi=_env_int("OCR_DPI", 200),
        ocr_language=os.environ.get("OCR_LANGUAGE", "eng"),
        ocr_max_pages=_env_int("OCR_MAX_PAGES", 100),
        ocr_workers=_env_int("OCR_WORKERS", 2),
        ocr_page_timeout_s=_env_int("OCR_PAGE_TIMEOUT_S", 60),
        groq_api_key=os.environ.get("GROQ_API_KEY") or None,
        groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        groq_reasoning_effort=_env_effort("GROQ_REASONING_EFFORT", "medium"),
        groq_max_tokens=_env_int("GROQ_MAX_TOKENS", 8192),
        groq_timeout_s=_env_int("GROQ_TIMEOUT_S", 60),
        max_history_turns=_env_int("MAX_HISTORY_TURNS", 8),
        cors_origins=tuple(
            o.strip()
            for o in os.environ.get("CORS_ORIGINS", "http://localhost:8501").split(",")
            if o.strip()
        ),
    )
