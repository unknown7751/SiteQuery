"""Custom CSS: a modern SaaS look layered on top of Streamlit."""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --brand: #ff6b1a;
  --brand-dark: #e5560a;
  --brand-soft: #fff1e8;
  --ink: #0f172a;
  --ink-2: #475569;
  --ink-3: #94a3b8;
  --surface: #ffffff;
  --canvas: #f6f7fb;
  --border: #e5e7ef;
  --ok: #059669;
  --ok-soft: #d1fae5;
  --warn: #b45309;
  --warn-soft: #fef3c7;
  --radius: 14px;
  --shadow-sm: 0 1px 2px rgba(15, 23, 42, .06);
  --shadow-md: 0 8px 24px rgba(15, 23, 42, .08);
}

html, body, [class*="css"], .stMarkdown, button, input, textarea {
  font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important;
}
.stApp { background: var(--canvas); color: var(--ink); }

/* --- hide Streamlit chrome ------------------------------------------------------ */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
.stDeployButton { display: none !important; visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 880px; padding-bottom: 7rem; }

/* --- sidebar --------------------------------------------------------------------- */
section[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
.brand { display: flex; align-items: center; gap: .7rem; margin-bottom: .2rem; }
.brand-logo {
  width: 38px; height: 38px; border-radius: 11px; display: grid; place-items: center;
  background: linear-gradient(135deg, var(--brand), #ff9a4d); color: #fff;
  font-weight: 700; font-size: 1.1rem; box-shadow: 0 6px 16px rgba(255, 107, 26, .35);
}
.brand-name { font-weight: 700; font-size: 1.05rem; line-height: 1.1; color: var(--ink); }
.brand-sub { font-size: .74rem; color: var(--ink-3); }
.section-label {
  text-transform: uppercase; letter-spacing: .08em; font-size: .68rem; font-weight: 600;
  color: var(--ink-3); margin: 1.3rem 0 .5rem;
}

/* status pill */
.pill {
  display: inline-flex; align-items: center; gap: .4rem; padding: .22rem .65rem;
  border-radius: 999px; font-size: .72rem; font-weight: 600;
}
.pill::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.pill.ok { background: var(--ok-soft); color: var(--ok); }
.pill.warn { background: var(--warn-soft); color: var(--warn); }

/* stat tiles */
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: .6rem; }
.stat {
  background: var(--canvas); border: 1px solid var(--border); border-radius: 12px;
  padding: .65rem .8rem;
}
.stat b { display: block; font-size: 1.25rem; color: var(--ink); }
.stat span { font-size: .72rem; color: var(--ink-2); }

/* document cards */
.doc-card {
  border: 1px solid var(--border); border-radius: 12px; padding: .6rem .75rem;
  background: var(--surface); box-shadow: var(--shadow-sm); margin-bottom: .1rem;
}
.doc-name {
  font-weight: 600; font-size: .85rem; color: var(--ink); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap;
}
.doc-meta { font-size: .72rem; color: var(--ink-2); margin-top: .15rem; }

/* file uploader as a drop zone */
[data-testid="stFileUploader"] section {
  border: 1.5px dashed #cbd5e1; border-radius: var(--radius); background: var(--canvas);
  padding: 1rem; transition: border-color .15s, background .15s;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--brand); background: var(--brand-soft); }

/* buttons */
.stButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
  border-radius: 10px; font-weight: 600; transition: all .15s;
}
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, var(--brand), var(--brand-dark)); border: 0; color: #fff;
  box-shadow: 0 6px 16px rgba(255, 107, 26, .3);
}
button[kind="primary"]:hover { transform: translateY(-1px); filter: brightness(1.05); }

/* --- hero / header -------------------------------------------------------------- */
.hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #3b2a20 100%);
  color: #fff; border-radius: 20px; padding: 1.7rem 1.9rem; margin-bottom: 1.2rem;
  box-shadow: var(--shadow-md); position: relative; overflow: hidden;
}
.hero::after {
  content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
  background: radial-gradient(circle, rgba(255, 107, 26, .45), transparent 70%);
}
.hero h1 { font-size: 1.55rem; font-weight: 700; margin: 0 0 .35rem; color: #fff; }
.hero p { margin: 0; color: #cbd5e1; font-size: .95rem; max-width: 560px; }

.topbar {
  display: flex; justify-content: space-between; align-items: center; margin-bottom: .8rem;
}
.topbar h2 { font-size: 1.05rem; margin: 0; font-weight: 700; color: var(--ink); }
.topbar small { color: var(--ink-2); }

/* --- chat ----------------------------------------------------------------------- */
[data-testid="stChatMessage"] {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 1rem 1.1rem; box-shadow: var(--shadow-sm); margin-bottom: .8rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--brand-soft); border-color: #ffd9c0;
}
[data-testid="stChatMessage"] p { line-height: 1.6; color: var(--ink); }

[data-testid="stChatInput"] {
  border-radius: 16px; border: 1px solid var(--border); box-shadow: var(--shadow-md);
  background: var(--surface);
}
[data-testid="stChatInput"]:focus-within { border-color: var(--brand); }

/* sources */
.src-label {
  text-transform: uppercase; letter-spacing: .08em; font-size: .68rem; font-weight: 600;
  color: var(--ink-3); margin: .9rem 0 .45rem;
}
.src {
  border: 1px solid var(--border); border-left: 3px solid var(--brand); border-radius: 10px;
  padding: .6rem .8rem; margin-bottom: .55rem; background: var(--canvas);
  scroll-margin-top: 5rem; transition: background .2s, box-shadow .2s;
}
.src-head { display: flex; justify-content: space-between; gap: .5rem; font-size: .8rem; }
.src-head b { color: var(--ink); }
.src-head > span { color: var(--ink-2); white-space: nowrap; }
.src-snippet {
  font-size: .8rem; color: var(--ink-2); margin-top: .3rem; line-height: 1.5;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
/* Citation links set the URL hash; :target expands and highlights the matching card. */
.src:target {
  background: var(--brand-soft); border-color: var(--brand);
  box-shadow: 0 0 0 3px rgba(255, 107, 26, .22);
}
.src:target .src-snippet { display: block; -webkit-line-clamp: unset; overflow: visible; color: var(--ink); }
.chip {
  display: inline-block; background: var(--brand-soft); color: var(--brand-dark);
  border-radius: 999px; padding: .05rem .5rem; font-size: .72rem; font-weight: 700;
  margin-right: .4rem;
}
a.cite {
  display: inline-block; background: var(--brand-soft); color: var(--brand-dark) !important;
  border-radius: 6px; padding: 0 .3rem; font-size: .8em; font-weight: 700;
  text-decoration: none !important; line-height: 1.5; margin: 0 .05rem;
}
a.cite:hover, a.cite:focus-visible { background: var(--brand); color: #fff !important; }
.badge {
  display: inline-block; border-radius: 6px; padding: .02rem .4rem; margin-left: .35rem;
  font-size: .68rem; font-weight: 600; border: 1px solid transparent; vertical-align: 1px;
}
.badge.native { background: var(--canvas); color: var(--ink-2); border-color: var(--border); }
.badge.ocr { background: var(--warn-soft); color: var(--warn); border-color: #fcd34d; }

[data-testid="stExpander"] {
  border: 1px solid var(--border); border-radius: 12px; background: var(--surface);
}
</style>
"""
