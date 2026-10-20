"""Domain exceptions raised by services and mapped to HTTP errors in routes."""


class AppError(Exception):
    """Base class; `status_code` is the HTTP status the route layer should return."""

    status_code: int = 500

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


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
