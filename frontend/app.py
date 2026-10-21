"""SiteQuery Document & RFI Assistant — Streamlit UI (layout and chat flow only)."""

import re
from html import escape
from typing import Any

import streamlit as st

import api_client as api
from styles import CSS

st.set_page_config(
    page_title="SiteQuery | Document & RFI Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- session state --------------------------------------------------------------------

MESSAGES = "messages"  # list[{"role", "content", "sources"}]
DOCS = "docs"  # cached list[DocumentItem] from the API
HEALTH = "health"  # api.Health | None
API_ERROR = "api_error"  # last connectivity error message | None
UPLOAD_KEY = "upload_key"  # rotated to reset the file_uploader
SELECTED = "selected_docs"  # document ids the chat is scoped to (empty = all)
TOP_K = "top_k"
PENDING = "pending_prompt"  # prompt queued by a suggestion button

SUGGESTIONS: list[str] = [
    "Summarize the open RFIs and who needs to respond",
    "What does the spec say about concrete curing requirements?",
    "List any conflicts between the drawings notes and specifications",
    "What are the submittal requirements for structural steel?",
]


def init_state() -> None:
    defaults: dict[str, Any] = {
        MESSAGES: [],
        DOCS: [],
        HEALTH: None,
        API_ERROR: None,
        UPLOAD_KEY: 0,
        SELECTED: [],
        TOP_K: 5,
        PENDING: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state[HEALTH] is None and st.session_state[API_ERROR] is None:
        refresh_server_state()


def refresh_server_state() -> None:
    try:
        st.session_state[HEALTH] = api.get_health()
        st.session_state[DOCS] = api.list_documents()
        st.session_state[API_ERROR] = None
    except api.ApiError as exc:
        st.session_state[HEALTH] = None
        st.session_state[API_ERROR] = exc.message


# ---- callbacks ------------------------------------------------------------------------


def queue_prompt(prompt: str) -> None:
    st.session_state[PENDING] = prompt


def clear_chat() -> None:
    st.session_state[MESSAGES] = []


def remove_document(doc_id: str) -> None:
    try:
        api.delete_document(doc_id)
    except api.ApiError as exc:
        st.toast(f"Could not delete: {exc.message}", icon="⚠️")
        return
    st.session_state[SELECTED] = [d for d in st.session_state[SELECTED] if d != doc_id]
    refresh_server_state()
    st.toast("Document removed", icon="🗑️")


# ---- renderers ------------------------------------------------------------------------


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


CITATION = re.compile(r"\[(\d{1,2})\]")


def link_citations(text: str, sources: list[dict[str, Any]], msg_id: int) -> str:
    """Turn `[n]` markers into in-page links to that message's source cards.

    Pure HTML/CSS (no JS): the link sets the URL hash and the card's `:target` style
    expands and highlights it. `<` is escaped first because document text can reach the answer.
    """
    known = {s["index"] for s in sources}

    def link(match: re.Match[str]) -> str:
        n = int(match.group(1))
        if n not in known:
            return match.group(0)
        return (
            f'<a class="cite" href="#src-{msg_id}-{n}" target="_self" '
            f'title="Show source [{n}]">[{n}]</a>'
        )

    return CITATION.sub(link, text.replace("<", "&lt;"))


def source_card(s: dict[str, Any], msg_id: int) -> str:
    page = f" · p. {s['page']}" if s["page"] else ""
    if s["ocr"]:
        badge = '<span class="badge ocr" title="Text recovered by OCR; may contain errors">🔍 OCR</span>'
    else:
        badge = '<span class="badge native" title="Text read directly from the file">📄 Native text</span>'
    if s["rerank_score"] is not None:
        score = (
            f'<span title="Cross-encoder relevance. Hybrid rank agreement: {s["relevance"]:.0%}">'
            f'Rerank {s["rerank_score"]:.0%}</span>'
        )
    else:
        score = f'<span title="Hybrid rank agreement">Match {s["relevance"]:.0%}</span>'
    # <details> makes the whole card click-to-expand without JS; `[open]` lifts the snippet clamp.
    return (
        f'<details class="src" id="src-{msg_id}-{s["index"]}"><summary>'
        f'<div class="src-head"><div><span class="chip">[{s["index"]}]</span>'
        f'<b>{escape(s["filename"])}</b>{page} {badge}</div>{score}</div>'
        f'<div class="src-snippet">{escape(s["snippet"])}</div></summary></details>'
    )


def render_sources(sources: list[dict[str, Any]], msg_id: int) -> None:
    if not sources:
        return
    cards = "".join(source_card(s, msg_id) for s in sources)
    st.markdown(
        f'<div class="src-label">📎 {len(sources)} sources · click a card or citation to expand</div>{cards}',
        unsafe_allow_html=True,
    )


def render_assistant_message(content: str, sources: list[dict[str, Any]], msg_id: int) -> None:
    st.markdown(link_citations(content, sources, msg_id), unsafe_allow_html=True)
    render_sources(sources, msg_id)


def render_sidebar() -> None:
    health: api.Health | None = st.session_state[HEALTH]
    with st.sidebar:
        st.markdown(
            """<div class="brand"><div class="brand-logo">S</div>
            <div><div class="brand-name">SiteQuery</div>
            <div class="brand-sub">Document &amp; RFI Assistant</div></div></div>""",
            unsafe_allow_html=True,
        )

        if st.session_state[API_ERROR]:
            st.markdown('<span class="pill warn">API offline</span>', unsafe_allow_html=True)
            st.error(st.session_state[API_ERROR])
            st.button("Retry connection", on_click=refresh_server_state, use_container_width=True)
            return

        assert health is not None
        if health.llm_configured:
            st.markdown(f'<span class="pill ok">Connected · {escape(health.model)}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill warn">No API key — Q&amp;A disabled</span>', unsafe_allow_html=True)

        st.markdown(
            f"""<div class="stats" style="margin-top:.9rem">
            <div class="stat"><b>{health.documents}</b><span>Documents</span></div>
            <div class="stat"><b>{health.chunks}</b><span>Indexed chunks</span></div></div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label">Upload documents</div>', unsafe_allow_html=True)
        files = st.file_uploader(
            "RFIs, specifications, submittals",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state[UPLOAD_KEY]}",
            label_visibility="collapsed",
        )
        if files and st.button(
            f"Index {len(files)} document{'s' if len(files) > 1 else ''}",
            type="primary",
            use_container_width=True,
        ):
            ingest_files(files)

        st.markdown('<div class="section-label">Library</div>', unsafe_allow_html=True)
        docs: list[api.DocumentItem] = st.session_state[DOCS]
        if not docs:
            st.caption("No documents yet. Upload your first RFI or spec above.")
        for doc in docs:
            info = [human_size(doc.size_bytes), f"{doc.chunks} chunks"]
            if doc.pages:
                info.insert(1, f"{doc.pages} pages")
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            c1.markdown(
                f"""<div class="doc-card"><div class="doc-name" title="{escape(doc.filename)}">
                {escape(doc.filename)}</div><div class="doc-meta">{' · '.join(info)}</div></div>""",
                unsafe_allow_html=True,
            )
            c2.button("✕", key=f"del_{doc.id}", help="Remove document", on_click=remove_document, args=(doc.id,))

        if docs:
            st.markdown('<div class="section-label">Search scope</div>', unsafe_allow_html=True)
            st.multiselect(
                "Limit questions to",
                options=[d.id for d in docs],
                format_func=lambda i: next((d.filename for d in docs if d.id == i), i),
                key=SELECTED,
                placeholder="All documents",
                label_visibility="collapsed",
            )

        st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
        st.slider("Passages retrieved per question", 1, 12, key=TOP_K)
        left, right = st.columns(2)
        left.button("Refresh", on_click=refresh_server_state, use_container_width=True)
        right.button("Clear chat", on_click=clear_chat, use_container_width=True)


def ingest_files(files: list[Any]) -> None:
    ok = 0
    with st.status("Indexing documents…", expanded=True) as status:
        for f in files:
            st.write(f"Processing **{f.name}**")
            try:
                api.upload_document(f.name, f.getvalue(), f.type)
                ok += 1
            except api.ApiError as exc:
                st.write(f"⚠️ {f.name}: {exc.message}")
        status.update(
            label=f"Indexed {ok} of {len(files)} document(s)",
            state="complete" if ok == len(files) else "error",
        )
    if ok:
        st.session_state[UPLOAD_KEY] += 1  # resets the uploader
        refresh_server_state()
        st.toast(f"Indexed {ok} document(s)", icon="✅")
        st.rerun()


def render_empty_state() -> None:
    st.markdown(
        """<div class="hero"><h1>Ask your project documents anything.</h1>
        <p>Upload RFIs and specifications, then get answers with citations back to the exact
        document and page — so your team spends less time searching and more time building.</p></div>""",
        unsafe_allow_html=True,
    )
    if not st.session_state[DOCS]:
        st.info("Start by uploading a document from the sidebar.", icon="📄")
        return
    st.markdown('<div class="section-label">Try asking</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        cols[i % 2].button(
            suggestion, key=f"sg_{i}", on_click=queue_prompt, args=(suggestion,), use_container_width=True
        )


def answer_question(prompt: str) -> None:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state[MESSAGES]]
    st.session_state[MESSAGES].append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🏗️"):
        try:
            with st.spinner("Searching documents and drafting an answer…"):
                answer = api.ask(
                    prompt,
                    history,
                    st.session_state[SELECTED] or None,
                    st.session_state[TOP_K],
                )
        except api.ApiError as exc:
            st.error(exc.message)
            st.session_state[MESSAGES].pop()  # drop the unanswered question
            return
        sources = [vars_of(s) for s in answer.sources]
        render_assistant_message(answer.text, sources, len(st.session_state[MESSAGES]))
    st.session_state[MESSAGES].append({"role": "assistant", "content": answer.text, "sources": sources})


def vars_of(source: api.Source) -> dict[str, Any]:
    return {
        "index": source.index,
        "filename": source.filename,
        "page": source.page,
        "snippet": source.snippet,
        "relevance": source.relevance,
        "rerank_score": source.rerank_score,
        "ocr": source.ocr,
    }


# ---- page -----------------------------------------------------------------------------


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    render_sidebar()

    if st.session_state[API_ERROR]:
        st.error("The backend API is unreachable. Start it and click **Retry connection**.")
        return

    # chat_input is pinned to the bottom by Streamlit, so reading it first is safe and lets
    # the hero/empty state be skipped on the same run that submits the first question.
    prompt = st.chat_input("Ask about your RFIs and specifications…") or st.session_state.pop(
        PENDING, None
    )

    messages: list[dict[str, Any]] = st.session_state[MESSAGES]
    if not messages and not prompt:
        render_empty_state()
    else:
        scope = len(st.session_state[SELECTED]) or "all"
        st.markdown(
            f'<div class="topbar"><h2>Conversation</h2><small>Searching {scope} document(s)</small></div>',
            unsafe_allow_html=True,
        )
    for msg_id, m in enumerate(messages):
        with st.chat_message(m["role"], avatar="🧑‍💼" if m["role"] == "user" else "🏗️"):
            if m["role"] == "assistant":
                render_assistant_message(m["content"], m["sources"], msg_id)
            else:
                st.markdown(m["content"])

    if prompt:
        answer_question(prompt)


main()
