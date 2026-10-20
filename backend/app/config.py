"""Application settings, read once from environment variables."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    max_upload_mb: int
    allowed_extensions: frozenset[str]
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    default_top_k: int
    anthropic_api_key: str | None
    anthropic_model: str
    anthropic_effort: str
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
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5"),
        anthropic_effort=os.environ.get("ANTHROPIC_EFFORT", "medium"),
        max_history_turns=_env_int("MAX_HISTORY_TURNS", 8),
        cors_origins=tuple(
            o.strip()
            for o in os.environ.get("CORS_ORIGINS", "http://localhost:8501").split(",")
            if o.strip()
        ),
    )
