---
paths:
  - "backend/**/*.py"
---

# Backend rules (FastAPI)

## Async everywhere
- All endpoints are `async def`. Never block the event loop: FAISS search/add/persist,
  embedding, PDF/DOCX parsing and file hashing run via `asyncio.to_thread`.
- Use LangChain's `ChatGroq` (built in `services/llm.py`) and `await llm.ainvoke(...)` for
  LLM calls. One shared client created in the lifespan.
- Create heavyweight objects (embeddings, FAISS store, cross-encoder reranker, LLM client) once in the
  `lifespan` handler, store on `app.state`, expose via `Depends` providers.

## File upload handling (be paranoid)
- Validate in this order, failing fast with the right status code:
  1. extension allow-list → `415`
  2. non-empty and within `MAX_UPLOAD_MB`, enforced WHILE streaming in chunks (never
     `await file.read()` the whole body) → `413` / `400`
  3. magic bytes match the extension (`%PDF-`, `PK` for DOCX, decodable text) → `415`
  4. parsing yields extractable text (scanned/empty PDFs) → `422`
- Never use the client filename as a path. Sanitize it and store as `<uuid>_<safe-name>`.
- Dedupe by SHA-256: re-uploading identical content returns `409` with the existing doc id.
- Any failure after bytes hit disk must delete the partial file and roll back index/registry
  changes (use try/except with cleanup, not hope).
- Always `await upload.close()` in `finally`.

## Error handling
- Services raise domain exceptions (`app/errors.py`); routes translate them to
  `HTTPException` with a clear, non-leaky `detail`. Unexpected errors → logged with
  `logger.exception` and returned as a generic `500`.
- LLM errors map explicitly (`groq` exceptions): missing key → `503`, `RateLimitError` → `429`
  (forward `Retry-After`), `AuthenticationError` / `PermissionDeniedError` → `500` (bad server
  key), `APIConnectionError` / `APIStatusError` → `502`. Catch the most specific class first.
- Endpoints declare `response_model`, status codes, and `responses=` for error cases.

## FAISS / vector logic separation
- All FAISS and embedding code lives in `backend/app/vectorstore/`. Nothing outside that
  package imports `faiss` or `langchain_community.vectorstores`.
- Routes never touch the index; they call `services/ingestion.py` and `services/rag.py`,
  which depend on the `VectorStore` interface (`vectorstore/base.py`) — swapping FAISS for
  pgvector/Qdrant later must not touch routes or services.
- The store owns its lock and persistence: embed OUTSIDE the lock, mutate under it, then persist.
  Never overwrite index files in place: write generation N+1 (`index.<n>.faiss/.pkl`), fsync the
  files and the directory, then flip the `current` pointer with `fsutil.atomic_write_bytes`.
  If a write fails, reload the last durable generation so memory never runs ahead of disk.
  Chunk ids are `"<doc_id>:<n>"` so deletes are exact; `delete` ignores ids that are not indexed.
- Anything else that must survive a crash (e.g. the document manifest) is written with
  `fsutil.atomic_write_bytes`, never `write_text` + `replace`.
- Embedding model name and cache dir come from settings; keep the cache on the mounted volume.
  The model name and vector dimension are persisted in `current`; startup refuses an index built
  with a different model (`EmbeddingMismatchError`) or one that is corrupt (`IndexCorruptedError`).
- Startup order in `lifespan`: registry -> `FaissVectorStore` -> `reconcile_index` -> `HybridVectorStore`
  (so BM25 is built only from reconciled chunks). A corrupt manifest is quarantined to
  `documents.json.corrupt` and aborts startup (`ManifestCorruptedError`); never reset it to empty.

## Loaders
- One loader class per file type in `backend/app/loaders/`, registered by extension in
  `registry.py`. Loaders return LangChain `Document`s with normalised metadata (`page` is
  1-indexed or absent). Adding a format = add a loader + one registry line.

## Typing & style
- Strict type hints, Pydantic v2 models for all request/response bodies, `from __future__`
  not required on 3.11+. Use `logging.getLogger(__name__)`; no `print`.
