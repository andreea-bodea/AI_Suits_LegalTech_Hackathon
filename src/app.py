import streamlit as st
import re
from io import BytesIO

import docx  
from PyPDF2 import PdfReader

from graph_agent import create_agent

st.set_page_config(page_title="Clause Analyzer", layout="wide")

st.title("📝 Rental Agreement Clause Review")

# 1) File uploader
uploaded = st.file_uploader("Upload your lease (PDF or DOCX)", 
                            type=["pdf", "docx"])
clause_dict = {}

if uploaded:
    # read full text
    if uploaded.type == "application/pdf":
        reader = PdfReader(BytesIO(uploaded.getvalue()))
        full_text = "\n\n".join(page.extract_text() for page in reader.pages)
    else:  # assume .docx
        doc = docx.Document(BytesIO(uploaded.getvalue()))
        full_text = "\n\n".join(p.text for p in doc.paragraphs)
    
    # split into clauses: look for '§ <number>' headings
    # this regex captures the section marker + its text up to the next '§'
    raw_sections = re.split(r"(§\s*\d+)", full_text)
    # raw_sections will be ["", "§ 1", "text…", "§ 2", "text…", …]
    for i in range(1, len(raw_sections), 2):
        sec_id = raw_sections[i].strip()           # e.g. "§ 1"
        sec_text = raw_sections[i+1].strip()       # the paragraph(s)
        clause_dict[sec_id] = sec_text
    
    if not clause_dict:
        st.warning("No clauses found. Make sure your document uses ‘§ <number>’ headings.")
    else:
        # 3) let user pick which clause to analyze
        choice = st.selectbox("Select clause to analyze", list(clause_dict.keys()))
        st.markdown(f"**{choice}**")
        st.write(clause_dict[choice])

        # 4) run analysis
        if st.button("🔍 Analyze selected clause"):
            agent = create_agent()
            with st.spinner("Analyzing…"):
                result = agent.run({
                    "clause_id": choice,
                    "text": clause_dict[choice]
                })
            st.subheader("Results")
            st.write(result)
else:
    st.info("Upload a PDF or DOCX rental agreement to get started.")
