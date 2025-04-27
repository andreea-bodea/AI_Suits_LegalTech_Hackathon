# app.py ---------------------------------------------------------------

import re
from io import BytesIO

import streamlit as st
import docx
from PyPDF2 import PdfReader
from streamlit_js_eval import streamlit_js_eval  # <- NEW

from graph_agent import create_agent

# ──────────────────────────────────────────────────────────────────────
# Page chrome
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Clause Analyzer", layout="wide")
st.title("📝 Rental Agreement Clause Review")

uploaded = st.file_uploader(
    "Upload your lease (PDF or DOCX)", type=["pdf", "docx"]
)
if not uploaded:
    st.info("Upload a PDF or DOCX rental agreement to get started.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────
# 1. Extract raw text
# ──────────────────────────────────────────────────────────────────────

def paragraphs_with_bullets(document):
    for para in document.paragraphs:
        if para._p.pPr is not None and para._p.pPr.numPr is not None:
            # it’s in a numbered/bulleted list
            level = para._p.pPr.numPr.ilvl.val  # 0, 1, 2 …
            bullet = "•" if level % 2 == 0 else "◦"
            txt = f"{bullet} {para.text}"
        else:
            txt = para.text
        if txt.strip():              # skip empty list style carriers
            yield txt

if uploaded.type == "application/pdf":
    reader = PdfReader(BytesIO(uploaded.getvalue()))
    full_text = "\n\n".join(p.extract_text() for p in reader.pages)
else:
    doc = docx.Document(BytesIO(uploaded.getvalue()))
    full_text = "\n".join(paragraphs_with_bullets(doc))
    full_text = re.sub(r"\n{2,}", "\n", full_text)   # collapse duplicates
    full_text = re.sub(r"(?m)^§", "\n§", full_text)  # <── add this line

# ──────────────────────────────────────────────────────────────────────
# 2. Detect clause headings that start a line with “§”
# ──────────────────────────────────────────────────────────────────────
heading_re = re.compile(r"(?m)^§\s*\d+[a-zA-Z]?\b[^\n]*")
matches = list(heading_re.finditer(full_text))
clauses: dict[str, str] = {}
for idx, m in enumerate(matches):
    body_start = m.end()
    body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
    clauses[m.group(0).strip()] = full_text[body_start:body_end].strip()

if not clauses:
    st.warning("No clauses found. Make sure headings start their own line with ‘§’.")
    st.stop()

# initialise session key
st.session_state.setdefault("active_clause", next(iter(clauses)))

# ──────────────────────────────────────────────────────────────────────
# Helper to wrap chosen clause in yellow highlight
# ──────────────────────────────────────────────────────────────────────
def preview_html(text: str, target: str | None) -> str:
    if not target or target not in clauses:
        return text.replace("\n", "<br>")
    pat = re.escape(target) + r".*?" + re.escape(clauses[target])
    block = re.search(pat, text, re.S)
    if not block:
        return text.replace("\n", "<br>")
    highlighted = (
        text[: block.start()]
        + "<a name='spot'></a>"
        + f"<span style='background:#fff4b2;padding:2px'>{block.group(0)}</span>"
        + text[block.end() :]
    )
    return highlighted.replace("\n", "<br>")

# ──────────────────────────────────────────────────────────────────────
# 3. Two-column layout
# ──────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1])

# LEFT ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
with left_col:
    st.subheader("📄 Lease preview")
    st.markdown(
        preview_html(full_text, st.session_state["active_clause"])
        + """
        <script>
          const anchor = document.querySelector("a[name='spot']");
          if (anchor) { anchor.scrollIntoView({behavior:'smooth', block:'center'}); }
        </script>
        """,
        unsafe_allow_html=True,
    )

# RIGHT ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
with right_col:
    st.subheader("💡 AI suggestions by clause")

    # cache the LangGraph runner once per session
    @st.cache_resource(hash_funcs={type(create_agent()): lambda _: None})
    def get_agent():
        return create_agent()

    agent = get_agent()

    for heading, body in clauses.items():
        with st.expander(heading, expanded=False):
            with st.spinner("Analyzing…"):
                result = agent.run({"clause_id": heading, "text": body})
            st.markdown("**Suggested improvement:**")
            st.write(result)

    # ──────────────────────
    # JS listener (streamlit-js-eval)
    # ──────────────────────
    clicked = streamlit_js_eval(
        js_expressions="""
        (() => {
          // attach at most once
          if (!window.__cl_listener_added__) {
            window.__cl_listener_added__ = true;

            // Streamlit root doc lives in parent
            const headers = parent.document.querySelectorAll('[data-testid="stExpanderHeader"]');
            headers.forEach(h => {
              if (!h.dataset.listenerAttached) {
                h.dataset.listenerAttached = 'yes';
                h.addEventListener('click', () => {
                  const title = h.innerText.split('\\n')[0];
                  Streamlit.setComponentValue(title);   // send back to Python
                });
              }
            });
          }
        })();
        """,
        want_output=True,
        key="expander_click",
    )

    # if JS returned something → update highlight + rerun
    if clicked and clicked != st.session_state["active_clause"]:
        st.session_state["active_clause"] = clicked
        st.experimental_rerun()