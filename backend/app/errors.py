"""Domain exceptions raised by services and mapped to HTTP errors in routes."""

from pathlib import Path


class AppError(Exception):
    """Base class; `status_code` is the HTTP status the route layer should return."""

    status_code: int = 500

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class ManifestCorruptedError(Exception):
    """The document manifest is unreadable. Raised at startup, never mapped to HTTP.

    Deliberately not an `AppError`: starting with an empty library would orphan every
    indexed chunk, so the service must refuse to boot until a person looks at it.
    """

    def __init__(self, path: Path, quarantined_to: Path | None, cause: str) -> None:
        moved = (
            f"It was moved to {quarantined_to}."
            if quarantined_to is not None
            else "It could not be moved aside and is still in place."
        )
        super().__init__(
            f"Document manifest {path} is unreadable ({cause}). {moved} Restore it from a backup "
            f"or repair the quarantined copy and put it back at {path}, then restart."
        )
        self.path = path
        self.quarantined_to = quarantined_to


class UnsupportedFileTypeError(AppError):
    status_code = 415


class FileTooLargeError(AppError):
    status_code = 413


class InvalidFileError(AppError):
    """Empty or corrupt upload."""

    status_code = 400


class NoExtractableTextError(AppError):
    """The file parsed but contained no text (e.g. a scanned PDF without OCR)."""

    status_code = 422


class DuplicateDocumentError(AppError):
    status_code = 409

    def __init__(self, detail: str, existing_id: str) -> None:
        super().__init__(detail)
        self.existing_id = existing_id


class DocumentNotFoundError(AppError):
    status_code = 404


class LLMNotConfiguredError(AppError):
    status_code = 503
