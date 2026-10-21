"""Startup-time failures of the on-disk index. Not HTTP errors: they stop the service from
booting, because serving queries from a doubtful index is worse than not serving."""


class VectorStoreError(Exception):
    """Base class for index persistence problems."""


class IndexCorruptedError(VectorStoreError):
    """The persisted index is unreadable or internally inconsistent."""


class EmbeddingMismatchError(VectorStoreError):
    """The index was built with a different embedding model than the one configured."""
