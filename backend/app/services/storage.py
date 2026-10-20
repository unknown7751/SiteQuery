"""Streaming, validated persistence of uploaded files."""

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from backend.app.config import Settings
from backend.app.errors import FileTooLargeError, InvalidFileError, UnsupportedFileTypeError
from backend.app.loaders import get_loader

logger = logging.getLogger(__name__)

_CHUNK = 1024 * 1024
_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]+")


@dataclass(frozen=True, slots=True)
class StoredFile:
    doc_id: str
    path: Path
    original_name: str
    size_bytes: int
    sha256: str


def sanitize_filename(name: str) -> str:
    base = Path(name.replace("\\", "/")).name  # drop any client-supplied directories
    cleaned = _UNSAFE.sub("_", base).strip(" .") or "document"
    return cleaned[:120]


async def save_upload(upload: UploadFile, settings: Settings) -> StoredFile:
    """Validate and stream `upload` to disk. Leaves no partial file behind on failure."""
    original = sanitize_filename(upload.filename or "")
    suffix = Path(original).suffix.lower()
    if suffix not in settings.allowed_extensions:
        allowed = ", ".join(sorted(settings.allowed_extensions))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix or 'none'}'. Allowed: {allowed}."
        )
    loader = get_loader(original)

    doc_id = uuid.uuid4().hex
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.upload_dir / f"{doc_id}_{original}"
    digest = hashlib.sha256()
    size = 0

    try:
        with dest.open("wb") as out:
            while chunk := await upload.read(_CHUNK):
                if size == 0 and not loader.validate_magic(chunk[:1024]):
                    raise UnsupportedFileTypeError(
                        f"File content does not look like a valid '{suffix}' file."
                    )
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise FileTooLargeError(
                        f"File exceeds the {settings.max_upload_mb} MB upload limit."
                    )
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise InvalidFileError("The uploaded file is empty.")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    return StoredFile(
        doc_id=doc_id,
        path=dest,
        original_name=original,
        size_bytes=size,
        sha256=digest.hexdigest(),
    )
