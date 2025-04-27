flowchart TD
    %% ─────────────── UI LAYER ───────────────
    subgraph UI[Streamlit Front‑End]
        A[User uploads<br>PDF / DOCX]
        I[Suggestions panel]
        K[Retrieval‑QA chatbot]
    end

    %% ─────────────── EXTRACTION ───────────────
    A --> B[Text extraction<br>& clause detection]
    B --> C{Iterate clauses}

    %% ─────────────── AGENT GRAPH ───────────────
    subgraph G1[LangGraph – per clause]
        direction LR
        C --> D[ReadClause / summarise ✨]
        D --> E[RetrieveCaseLaw 🔍]
        D --> F[EvaluateRisk ⚖️]
        E --> F
        F --> G[SuggestImprovement 📝]
    end

    %% ─────────────── DATA FLOW ───────────────
    G --> H[Session‑state<br>suggestions_db]
    H --> I

    %% Chat RAG path
    H --> J[In‑memory Chroma<br>clause_suggestions]
    J --> K

    %% ─────────────── VECTOR STORES ───────────────
    subgraph VS[Persistent Chroma Stores]
        L(case_law_index)
    end
    E -. search .-> L

    %% Statute ingestion job
    subgraph Batch[Offline / CLI]
        M[provisions_ingest.py]
    end
    M -- add embeddings --> L