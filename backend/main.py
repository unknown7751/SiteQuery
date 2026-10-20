"""FastAPI entrypoint: lifespan wiring, error mapping, and thin async routes."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import groq
from fastapi import Depends, FastAPI, File, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.documents import Document

from backend.app.config import Settings, get_settings
from backend.app.errors import AppError, DuplicateDocumentError
from backend.app.schemas import (
    DocumentListResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RootResponse,
    UploadResponse,
)
from backend.app.services.ingestion import IngestionService
from backend.app.services.llm import build_llm
from backend.app.services.rag import RagService
from backend.app.services.registry import DocumentRegistry
from backend.app.vectorstore import (
    CrossEncoderReranker,
    FaissVectorStore,
    HybridVectorStore,
    Reranker,
    SearchHit,
    build_embeddings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _warmup_hit() -> SearchHit:
    return SearchHit(document=Document(page_content="warm-up"), relevance=0.0)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    for directory in (settings.upload_dir, settings.index_dir, settings.model_cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Model download / index load are blocking; keep them off the event loop.
    embeddings = await asyncio.to_thread(
        build_embeddings, settings.embedding_model, settings.model_cache_dir
    )
    await asyncio.to_thread(embeddings.embed_query, "warm-up")
    dense = await asyncio.to_thread(FaissVectorStore, settings.index_dir, embeddings)
    store = await asyncio.to_thread(
        HybridVectorStore, dense, settings.retrieval_candidates, settings.rrf_k
    )
    registry = DocumentRegistry(settings.index_dir / "documents.json")

    reranker: Reranker | None = None
    if settings.rerank_enabled:
        cross_encoder = await asyncio.to_thread(
            CrossEncoderReranker,
            settings.rerank_model,
            settings.model_cache_dir,
            settings.rerank_threshold,
        )
        await asyncio.to_thread(cross_encoder.rerank, "warm-up", [_warmup_hit()], 1)
        reranker = cross_encoder
    else:
        logger.info("Cross-encoder reranking disabled (RERANK_ENABLED=false)")

    llm = build_llm(settings)
    if llm is None:
        logger.warning("GROQ_API_KEY not set: uploads work, /query will return 503")

    app.state.settings = settings
    app.state.registry = registry
    app.state.store = store
    app.state.ingestion = IngestionService(settings, store, registry)
    app.state.rag = RagService(settings, store, llm, reranker)
    logger.info("Startup complete: %d vectors indexed", store.size)
    yield


app = FastAPI(
    title="vConstruct Document & RFI Assistant API",
    version="0.1.0",
    description="Upload construction documents and ask grounded, cited questions about them.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- dependency providers -------------------------------------------------------------


def get_ingestion(request: Request) -> IngestionService:
    return request.app.state.ingestion  # type: ignore[no-any-return]


def get_rag(request: Request) -> RagService:
    return request.app.state.rag  # type: ignore[no-any-return]


def get_registry(request: Request) -> DocumentRegistry:
    return request.app.state.registry  # type: ignore[no-any-return]


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


IngestionDep = Annotated[IngestionService, Depends(get_ingestion)]
RagDep = Annotated[RagService, Depends(get_rag)]
RegistryDep = Annotated[DocumentRegistry, Depends(get_registry)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]


# ---- error mapping --------------------------------------------------------------------


@app.exception_handler(AppError)
async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
    body: dict[str, object] = {"detail": exc.detail}
    if isinstance(exc, DuplicateDocumentError):
        body["existing_id"] = exc.existing_id
    return JSONResponse(status_code=exc.status_code, content=body)


# Starlette picks the handler for the most specific class in the exception's MRO, so the
# broad APIStatusError handler below only catches what the specific ones don't.


@app.exception_handler(groq.RateLimitError)
async def handle_rate_limit(_: Request, exc: groq.RateLimitError) -> JSONResponse:
    logger.warning("Groq rate limit: %s", exc)
    headers: dict[str, str] = {}
    retry_after = exc.response.headers.get("retry-after")
    if retry_after:
        headers["Retry-After"] = retry_after
    wait = f" Try again in about {retry_after} seconds." if retry_after else " Please retry shortly."
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": f"The language model is rate limited.{wait}"},
        headers=headers,
    )


@app.exception_handler(groq.AuthenticationError)
@app.exception_handler(groq.PermissionDeniedError)
async def handle_llm_auth(_: Request, exc: groq.APIStatusError) -> JSONResponse:
    # Server-side misconfiguration, not the caller's fault: hence 500, and name the fix.
    logger.error("Groq rejected the API key (%s): %s", exc.status_code, exc.message)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "The language model rejected the server's credentials. "
            "Check that GROQ_API_KEY is valid and has access to the configured model."
        },
    )


@app.exception_handler(groq.APIConnectionError)
async def handle_llm_connection(_: Request, exc: groq.APIConnectionError) -> JSONResponse:
    logger.error("Groq connection error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": "Could not reach the language model service."},
    )


@app.exception_handler(groq.APIStatusError)
async def handle_llm_status(_: Request, exc: groq.APIStatusError) -> JSONResponse:
    logger.error("Groq API error %s: %s", exc.status_code, exc.message)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": "The language model service returned an error."},
    )


@app.exception_handler(Exception)
async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error", exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


# ---- routes ---------------------------------------------------------------------------


@app.get("/", response_model=RootResponse, tags=["system"])
async def root() -> RootResponse:
    return RootResponse(status="API is running", service=app.title, docs="/docs")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request, registry: RegistryDep, rag: RagDep, settings: SettingsDep) -> HealthResponse:
    records = await asyncio.to_thread(registry.list)
    return HealthResponse(
        status="ok",
        documents=len(records),
        chunks=request.app.state.store.size,
        llm_configured=rag.llm_configured,
        model=settings.groq_model,
    )


@app.post(
    "/upload_document",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
    responses={
        400: {"description": "Empty or unreadable file"},
        409: {"description": "Identical document already uploaded"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported or mismatched file type"},
        422: {"description": "No extractable text (e.g. scanned PDF)"},
    },
)
async def upload_document(
    ingestion: IngestionDep, file: Annotated[UploadFile, File(description="PDF, DOCX, TXT or MD")]
) -> UploadResponse:
    info = await ingestion.ingest(file)
    return UploadResponse(
        document=info, message=f"Indexed '{info.filename}' into {info.chunks} searchable chunks."
    )


@app.get("/documents", response_model=DocumentListResponse, tags=["documents"])
async def list_documents(request: Request, registry: RegistryDep) -> DocumentListResponse:
    records = await asyncio.to_thread(registry.list)
    return DocumentListResponse(
        documents=[r.to_info() for r in records], total_chunks=request.app.state.store.size
    )


@app.delete(
    "/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["documents"],
    responses={404: {"description": "Document not found"}},
)
async def delete_document(doc_id: str, ingestion: IngestionDep) -> None:
    await ingestion.delete(doc_id)


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["chat"],
    responses={
        429: {"description": "Model rate limited (Retry-After header when provided)"},
        500: {"description": "Model rejected the server's GROQ_API_KEY"},
        502: {"description": "Model service unreachable or returned an error"},
        503: {"description": "GROQ_API_KEY not configured"},
    },
)
async def query(body: QueryRequest, rag: RagDep) -> QueryResponse:
    return await rag.answer(body)
