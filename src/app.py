# app.py  ────────────────────────────────────────────────────────────────────
"""Streamlit app to review rental–contract clauses and suggest improvements.

Features
~~~~~~~~
* **Incremental analysis** – a clause is analysed on first click; results are
  cached.
* **Vector‑store chatbot** – questions are answered from both the original
  clause text *and* the AI suggestions via an in‑memory Chroma index.
* **Collapsed preview expander**, status icons in the sidebar, smooth scroll to
  the active clause.
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Dict

import docx
import streamlit as st
from PyPDF2 import PdfReader

from graph_agent import create_agent
from suggestion_rag import answer_question  # ← NEW import

# ────────────────────────────────────────────────────────────────────────────
# Page‑level chrome & CSS helpers
# ────────────────────────────────────────────────────────────────────────────

def _add_global_css() -> None:
    st.markdown(
        """
        <style>
        /* wider reading area */
        .appview-container .main { max-width: 1450px; }
        /* modern font */
        * { font-family: 'Inter', sans-serif; }
        /* active clause button in sidebar – subtle bold */
        button.sidebar-active { font-weight: 600 !important; }
        /* gap between icon and text */
        button[data-testid="baseButton-element"] span:first-child { margin-right: .35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Clause Analyzer", layout="wide")
_add_global_css()

st.title("📝 Rental Agreement Clause Review")

# ────────────────────────────────────────────────────────────────────────────
# 1 ░ Upload – allow PDF or DOCX
# ────────────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader("Upload your rental contract (PDF or DOCX)", type=["pdf", "docx"])
if not uploaded:
    st.stop()

# ────────────────────────────────────────────────────────────────────────────
# 2 ░ Extract raw text while preserving bullet formatting
# ────────────────────────────────────────────────────────────────────────────

def _paragraphs_with_bullets(document: docx.document.Document):
    for para in document.paragraphs:
        if para._p.pPr is not None and para._p.pPr.numPr is not None:
            level = para._p.pPr.numPr.ilvl.val or 0
            bullet = "•" if level % 2 == 0 else "◦"
            txt = f"{bullet} {para.text}"
        else:
            txt = para.text
        if txt.strip():
            yield txt


if uploaded.type == "application/pdf":
    reader = PdfReader(BytesIO(uploaded.getvalue()))
    full_text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
else:
    doc = docx.Document(BytesIO(uploaded.getvalue()))
    full_text = "\n".join(_paragraphs_with_bullets(doc))
    full_text = re.sub(r"\n{2,}", "\n", full_text)
    full_text = re.sub(r"(?m)^§", "\n§", full_text)

# ────────────────────────────────────────────────────────────────────────────
# 3 ░ Detect clause headings that start with “§”
# ────────────────────────────────────────────────────────────────────────────

heading_re = re.compile(r"(?m)^§\s*\d+[a-zA-Z]?\b[^\n]*")
heading_matches = list(heading_re.finditer(full_text))
clauses: Dict[str, str] = {}
for idx, m in enumerate(heading_matches):
    body_start = m.end()
    body_end = heading_matches[idx + 1].start() if idx + 1 < len(heading_matches) else len(full_text)
    clauses[m.group(0).strip()] = full_text[body_start:body_end].strip()

if not clauses:
    st.warning("No clauses found. Make sure headings start their own line with ‘§’.")
    st.stop()

# State holders
st.session_state.setdefault("active_clause", next(iter(clauses)))
st.session_state.setdefault("suggestions_db", {})

# ────────────────────────────────────────────────────────────────────────────
# 4 ░ Helpers – preview HTML & analysis cache
# ────────────────────────────────────────────────────────────────────────────

def preview_html(text: str, anchor_clause: str | None) -> str:
    if not anchor_clause or anchor_clause not in clauses:
        return text.replace("\n", "<br>")

    pat = re.escape(anchor_clause)
    return re.sub(
        pat,
        lambda m: f"<a name='spot'></a>{m.group(0)}",
        text,
        count=1,
        flags=re.M,
    ).replace("\n", "<br>")


@st.cache_resource(hash_funcs={type(create_agent()): lambda _: None})
def _get_agent():
    return create_agent()

agent = _get_agent()

@st.cache_data(show_spinner=False)
def analyse_clause_once(clause_id: str, body: str) -> str:
    return agent.run({"clause_id": clause_id, "text": body})

# ────────────────────────────────────────────────────────────────────────────
# 5 ░ Sidebar navigation (buttons per clause)
# ────────────────────────────────────────────────────────────────────────────

st.sidebar.title("💡 AI suggestions")

for heading in clauses:
    analysed = heading in st.session_state["suggestions_db"]
    icon = "✅" if analysed else "⏳"

    if st.sidebar.button(f"{icon}  {heading}", key=heading, use_container_width=True):
        st.session_state["active_clause"] = heading
        st.rerun()

    # Toggle bold state via JS
    st.sidebar.markdown(
        f"""
        <script>
        const btn = window.parent.document.querySelector('button[data-testid="baseButton-element"][title="{heading}"]');
        if (btn) {{
            btn.classList.{ 'add' if heading == st.session_state['active_clause'] else 'remove' }('sidebar-active');
        }}
        </script>
        """,
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────────────────
# 6 ░ Main column – preview expander & suggestion card
# ────────────────────────────────────────────────────────────────────────────

with st.expander("📄 Rental Contract", expanded=False):
    st.markdown(preview_html(full_text, st.session_state["active_clause"]), unsafe_allow_html=True)
    st.markdown(
        """
        <script>
        const anchor = window.parent.document.querySelector('a[name="spot"]');
        if (anchor) { anchor.scrollIntoView({behavior:'smooth', block:'center'}); }
        </script>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
active_clause = st.session_state["active_clause"]

if active_clause not in st.session_state["suggestions_db"]:
    with st.spinner("Analyzing selected clause…"):
        st.session_state["suggestions_db"][active_clause] = analyse_clause_once(
            active_clause, clauses[active_clause]
        )

st.subheader(f"💡 AI suggestion for **{active_clause}**")
st.write(st.session_state["suggestions_db"][active_clause])

# ────────────────────────────────────────────────────────────────────────────
# 7 ░ Chatbot – vector‑store Q‑A
# ────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🤖 Ask the AI about these suggestions")

for msg in st.session_state.get("chat_history", []):
    st.chat_message(msg["role"]).write(msg["content"])

query = st.chat_input("Type a question about the proposed changes…")

if query:
    st.chat_message("user").write(query)
    st.session_state.setdefault("chat_history", []).append({"role": "user", "content": query})

    answer = answer_question(query, clauses, st.session_state["suggestions_db"])

    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})