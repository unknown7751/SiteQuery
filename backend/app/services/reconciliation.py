"""Startup cross-check of the vector index against the document manifest.

The two are written separately (index first, manifest second), so a crash in between leaves
"phantom" vectors: searchable chunks that belong to no listed document. This drops them, so
the keyword index built afterwards from the store covers only valid chunks.
"""

import logging
from dataclasses import dataclass, field

from backend.app.services.registry import DocumentRegistry
from backend.app.vectorstore import ReconcilableStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    phantoms_dropped: int = 0
    # Manifest records whose chunks are (partly) absent from the index, as (doc id, missing count).
    # Reported, never auto-deleted: removing a user's document is not ours to decide.
    incomplete_documents: list[tuple[str, int]] = field(default_factory=list)
    # True when the manifest was empty but the index was not, so nothing was dropped.
    skipped_empty_manifest: bool = False


def reconcile_index(store: ReconcilableStore, registry: DocumentRegistry) -> ReconcileResult:
    """Drop vectors the manifest does not know about; report manifest entries the index lost.

    Refuses to drop anything when the manifest lists no documents at all but the index has
    vectors: that is far more likely a deleted/replaced manifest than a crash on the very first
    upload, and wiping the whole index on that guess is not recoverable.
    """
    records = registry.list()
    indexed = store.chunk_ids()
    expected = {chunk_id for record in records for chunk_id in record.chunk_ids}

    phantoms = sorted(indexed - expected)
    skipped = False
    if phantoms and not records:
        logger.error(
            "Index holds %d vectors but the manifest lists no documents; leaving the index "
            "untouched. If the manifest was lost, restore it; if the index is junk, delete it.",
            len(phantoms),
        )
        phantoms, skipped = [], True
    elif phantoms:
        store.delete(phantoms)
        logger.warning("Dropped %d phantom vector(s) not present in the manifest", len(phantoms))

    incomplete = [
        (record.id, missing)
        for record in records
        if (missing := sum(1 for chunk_id in record.chunk_ids if chunk_id not in indexed)) > 0
    ]
    for doc_id, missing in incomplete:
        logger.warning("Manifest document %s is missing %d chunk(s) in the index", doc_id, missing)

    return ReconcileResult(len(phantoms), incomplete, skipped)
