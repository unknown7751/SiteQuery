---
paths:
  - "frontend/**/*.py"
---

# Frontend rules (Streamlit)

Always build top-notch, enterprise-grade Streamlit UIs. The bar is "modern B2B SaaS",
not "data-science demo".

## Visual design
- Inject custom CSS from `frontend/styles.py` via a single `st.markdown(..., unsafe_allow_html=True)`
  call per run. Define design tokens as CSS variables (`--brand`, `--surface`, `--border`,
  `--radius`, `--shadow`) and reuse them; no scattered inline styles.
- Consistent type scale (Inter or system stack), generous spacing, subtle borders and
  soft shadows, one brand accent colour. Hide Streamlit chrome (default menu/footer/deploy
  button) that breaks the product feel.
- Style the sidebar as a distinct management panel: brand header, section labels,
  document cards, status pills, and a clear primary action for uploads.
- Empty states are designed, not blank: show a welcome hero with suggested prompts.
- Support light backgrounds with readable contrast (WCAG AA); never rely on colour alone
  to convey status.

## Chat interface
- Use `st.chat_message` + `st.chat_input` for the conversation; custom avatars for user and
  assistant. Render assistant answers as markdown.
- Every assistant answer shows its sources (file, page, relevance, snippet) in a collapsed
  expander or chips beneath the message, so users can verify claims.
- Show progress for slow operations (`st.spinner` / `st.status`); never freeze silently.
- Provide a "Clear conversation" action and suggested follow-up prompts.

## State management
- Initialise ALL `st.session_state` keys in one `init_state()` function guarded by
  `if key not in st.session_state`. Keys are namespaced constants, not magic strings
  scattered around.
- Session state holds UI state only (messages, selected documents, cached document list,
  upload counter). Server state (documents, index) is fetched from the API and cached
  with a short TTL or an explicit refresh — never duplicated as truth.
- Callbacks (`on_click`/`on_change`) mutate state; avoid mutating state mid-render.
- Reset `st.file_uploader` via a rotating `key` after a successful upload.

## Architecture
- The UI talks to the backend ONLY through `frontend/api_client.py` over HTTP. Never
  import from `backend/`.
- API base URL comes from the `API_BASE_URL` env var (`http://api:8000` in Docker).
- Handle API failures gracefully: connection errors, 4xx validation messages, 5xx — show
  friendly `st.error`/`st.toast` text with the server's `detail` when available. No stack
  traces in the UI.
- Keep `app.py` as layout/flow only; extract reusable renderers into small functions with
  full type hints. Sibling imports (`from api_client import ...`) — Streamlit puts the
  script directory on `sys.path`.
