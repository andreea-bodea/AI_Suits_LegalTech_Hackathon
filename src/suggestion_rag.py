"""suggestion_rag.py
Vector‑store RAG helper for the clause‑suggestion chatbot.

It builds an **in‑memory Chroma index** that contains *both* the original
clause text and the AI‑generated suggestions, so follow‑up questions can pull
from either.

Usage (in app.py)
=================
```python
from suggestion_rag import answer_question
...
answer = answer_question(query, clauses, suggestions_db)
```

The function is stateless and quick because the data set is tiny (just the
clauses you’ve already opened).  If you later process hundreds of clauses you
can move the `build_index()` call into a `st.cache_data` wrapper.
"""

from __future__ import annotations

from typing import Dict, List

from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _make_documents(clauses: Dict[str, str], suggestions: Dict[str, str]) -> List[Document]:
    """Pack each clause + its suggestion into a single LangChain Document."""
    docs: List[Document] = []
    for heading, clause_body in clauses.items():
        suggestion_text = suggestions.get(heading, "")
        chunk = (
            f"Clause heading: {heading}\n\n"
            f"Original clause:\n{clause_body}\n\n"
            f"AI suggestions:\n{suggestion_text or '[no suggestion yet]'}"
        )
        docs.append(Document(page_content=chunk, metadata={"heading": heading}))
    return docs


def _build_index(clauses: Dict[str, str], suggestions: Dict[str, str]):
    """Return an in‑memory Chroma vector store for the provided data."""
    docs = _make_documents(clauses, suggestions)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # `persist_directory=None` ⇒ in‑memory index
    return Chroma.from_documents(docs, embeddings, collection_name="clause_suggestions")


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

def answer_question(question: str, clauses: Dict[str, str], suggestions: Dict[str, str], *, k: int = 4) -> str:
    """Return an answer drawn from the most relevant clause/suggestion chunks."""
    if not clauses:
        return "I’m sorry – there are no clauses loaded yet."

    # (Re)build index each call – for tens of clauses this is <100 ms.
    vect = _build_index(clauses, suggestions)
    retriever = vect.as_retriever(search_kwargs={"k": k})

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )

    return qa_chain.run(question).strip()