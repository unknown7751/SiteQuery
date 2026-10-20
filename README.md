# vConstruct Document & RFI Assistant

Upload RFIs and specifications, ask questions, get answers cited back to document and page.

```bash
cp .env.example .env        # set GROQ_API_KEY (Q&A needs it; uploads/indexing don't)
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
| `GET /`, `GET /health` | liveness; counts and whether an API key is configured |

## Retrieval and reranking

`/query` runs FAISS dense search and BM25 in parallel, fuses them with Reciprocal Rank Fusion, then
re-scores the top `RERANK_CANDIDATES` (20) with a local cross-encoder
(`Xenova/ms-marco-MiniLM-L-6-v2`, the ONNX build of `cross-encoder/ms-marco-MiniLM-L-6-v2`, via FastEmbed).
Only the best `top_k` (default 5) that score at least `RERANK_THRESHOLD` (default 0.10) go to Groq, so the LLM may
get fewer than `top_k` chunks, or none (the API then replies that nothing relevant was found, without an LLM call).
The MS-MARCO model can score casual paraphrases very low; lower the threshold (or set it to `0`) if you see
false "nothing found" answers. Each source returns:

| Field | Meaning |
| --- | --- |
| `relevance` | Normalised RRF rank agreement (0..1) |
| `rerank_score` | Cross-encoder relevance, sigmoid of the logit (0..1); `null` if reranking is off |
| `ocr` | `true` if the chunk text came from OCR rather than the PDF's text layer |

The first start also downloads the ~80 MB reranker into `data/models/`. Set `RERANK_ENABLED=false` to skip it.

In the chat UI, citation markers like **[2]** in an answer are links: clicking one scrolls to and expands the
matching source card, which shows its rerank score and a *Native text* / *OCR* badge.

## Evaluation

`eval/` holds a synthetic benchmark (4 construction documents, 10 question / ground-truth pairs).

```bash
pip install -r eval/requirements.txt
python -m eval.evaluate_rag --compare           # hybrid vs hybrid + reranker, writes eval/reports/latest.md
python -m eval.evaluate_rag --retrieval-only    # no LLM calls; hit@k and MRR only
# or inside the API container (uses the cached models and GROQ_API_KEY):
docker compose run --rm -v ./eval:/app/eval api python -m eval.evaluate_rag --compare
```

It indexes the sample documents into a temporary directory (your `data/` is untouched) and reports
**Faithfulness** (claims supported by context), **Answer relevance** (Ragas-style: embedding similarity
between the question and questions inferred from the answer), **Context precision** (average precision of
the retrieved ranking against the ground truth) and deterministic **hit@k / MRR**. The first three use
Groq as judge, so expect optimistic scores and a slow run on the free tier (429s are retried).
The benchmark is small (11 chunks), so treat it as a regression check, not proof of quality. Currently hybrid-only
scores hit@k 1.00 while hybrid + reranker + the 0.10 cutoff scores 0.90: the cross-encoder rates the casual
paraphrase Q10 near 0, so its chunk is dropped. Run with `RERANK_THRESHOLD=0` to see the reranker without the cutoff.

Local dev (Python 3.11+): `pip install -r backend/requirements.txt -r frontend/requirements.txt`,
then `uvicorn backend.main:app --reload` and `streamlit run frontend/app.py`.
See `CLAUDE.md` and `.claude/rules/` for conventions.
