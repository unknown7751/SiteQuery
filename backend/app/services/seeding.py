"""First-run seeding: index the bundled sample documents so a fresh install is queryable."""

import asyncio
import logging

from starlette.datastructures import UploadFile

from backend.app.config import Settings
from backend.app.errors import AppError
from backend.app.services.ingestion import IngestionService
from backend.app.services.registry import DocumentRegistry
from backend.app.vectorstore import VectorStore

logger = logging.getLogger(__name__)


async def seed_sample_data(
    settings: Settings,
    store: VectorStore,
    registry: DocumentRegistry,
    ingestion: IngestionService,
) -> int:
    """Ingest `settings.sample_data_dir` iff the index and manifest are both empty.

    Goes through `IngestionService.ingest` so seeded files get the same validation, hashing
    and rollback as user uploads. A bad sample file is logged and skipped; it must never
    stop the API from starting. Returns the number of documents ingested.
    """
    if not settings.seed_sample_data:
        return 0
    if store.size > 0 or await asyncio.to_thread(registry.list):
        return 0
    directory = settings.sample_data_dir
    if not directory.is_dir():
        logger.info("Sample data directory %s not found; skipping seed", directory)
        return 0

    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in settings.allowed_extensions
    )
    seeded = 0
    for path in files:
        try:
            with path.open("rb") as handle:
                await ingestion.ingest(UploadFile(file=handle, filename=path.name))
            seeded += 1
        except AppError as exc:
            logger.warning("Skipping sample file %s: %s", path.name, exc.detail)
    logger.info("Seeded %d sample document(s) from %s", seeded, directory)
    return seeded
