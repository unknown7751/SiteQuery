# vConstruct Document & RFI Assistant (Phase 1)

Upload RFIs and specifications, ask questions, get answers cited back to document and page.

```bash
cp .env.example .env        # set ANTHROPIC_API_KEY (Q&A needs it; uploads/indexing don't)
docker compose up --build   # UI http://localhost:8501 · API docs http://localhost:8000/docs
```

The first start downloads the local embedding model (~100 MB) into `data/models/`.
`data/uploads/` (files), `data/index/` (FAISS index + `documents.json`) and `data/models/`
are bind-mounted, so everything survives `docker compose down`.

| Endpoint | Purpose |
| --- | --- |
| `POST /upload_document` | multipart `file` (PDF, DOCX, TXT, MD) → parse, chunk, embed, index |
| `POST /query` | `{question, history?, top_k?, document_ids?}` → `{answer, sources[]}` |
| `GET /documents`, `DELETE /documents/{id}` | manage the library |
| `GET /health` | status, counts, whether an API key is configured |

Local dev (Python 3.11+): `pip install -r backend/requirements.txt -r frontend/requirements.txt`,
then `uvicorn backend.main:app --reload` and `streamlit run frontend/app.py`.
See `CLAUDE.md` and `.claude/rules/` for conventions.
