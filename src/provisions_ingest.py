# provisions_ingest.py
"""
Utility script to scrape statutory provisions from public web pages and append
(or create) a Chroma vector‑store that the ClauseGuard risk‑assessment engine
uses. **Run it with no arguments** to ingest a default set of statutes, or
provide custom `--urls` and/or `--persist_dir`.

Fix 2025‑04‑27
--------------
Recent versions of **langchain‑chroma** removed the `persist()` method from the
wrapper class.  The script now checks for it and falls back to
`vect._client.persist()` when necessary (or just skips the call when the store
is in‑memory).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────────────────────
# Environment & defaults
# ────────────────────────────────────────────────────────────────────────────

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    sys.exit("[ERROR] OPENAI_API_KEY not found – set it in your .env file.")

DEFAULT_URLS: List[str] = [
    # Mietrechtsgesetz (MRG)
    "https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer=10002531",
    # Allgemeines Bürgerliches Gesetzbuch (extracts)
    "https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer=10001622",
    # Konsumentenschutzgesetz (KSchG)
    "https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer=10007517",
    # Heizkostenabrechnungsgesetz (HeizKG)
    "https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer=10008184",
    # EU GDPR
    "https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng",
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ────────────────────────────────────────────────────────────────────────────
# Helpers – HTML → text
# ────────────────────────────────────────────────────────────────────────────

def _fetch_html(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

# ────────────────────────────────────────────────────────────────────────────
# Vector‑store ingestion
# ────────────────────────────────────────────────────────────────────────────

def _chunk_text(text: str, source_url: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " "]
    )
    for chunk in splitter.split_text(text):
        yield Document(page_content=chunk, metadata={"source": source_url})


def _safe_persist(vect: Chroma):
    """Persist the DB if the current langchain‑chroma version supports it."""
    if hasattr(vect, "persist"):
        # Old API (≤ 0.0.6)
        vect.persist()
    elif hasattr(vect, "_client") and hasattr(vect._client, "persist"):
        # Newer API: underlying chromadb client exposes persist()
        vect._client.persist()
    else:
        # In‑memory store (persist_directory=None) – nothing to do
        print("[INFO] Vector‑store lives only in memory – skipping persist().")


def ingest_urls(urls: List[str], persist_dir: str):
    print(f"[INFO] Initialising vector‑store in '{persist_dir}' …")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=OPENAI_API_KEY,
    )

    vect = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    new_docs = []
    for url in urls:
        try:
            print(f"[INFO] Fetching → {url}")
            html = _fetch_html(url)
            text = _clean_text(html)
            chunks = list(_chunk_text(text, url))
            print(f"        … extracted {len(chunks)} chunks")
            new_docs.extend(chunks)
        except Exception as exc:
            print(f"[WARN] Skipped {url} – {exc}")

    if not new_docs:
        print("[INFO] No new documents to add; exiting.")
        return

    print(f"[INFO] Adding {len(new_docs)} chunks to vector‑store …")
    vect.add_documents(new_docs)
    _safe_persist(vect)
    print("[SUCCESS] Provisions ingested and persisted.")

# ────────────────────────────────────────────────────────────────────────────
# CLI entry‑point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape legal provisions and embed them into a Chroma store.")
    parser.add_argument(
        "--persist_dir",
        default="./case_law_index",
        help="Path to Chroma persist directory (default: ./case_law_index)",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        default=DEFAULT_URLS,
        help="HTTP URLs to ingest (default: built‑in statutory list)",
    )
    args = parser.parse_args()

    ingest_urls(args.urls, args.persist_dir)