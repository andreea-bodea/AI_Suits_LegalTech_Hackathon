# graph_agent.py
"""
LangGraph workflow for the clause-analysis app.
✓ Uses StateGraph (fan-out support) and the current API (graph.invoke).
✓ Exposes a .run() wrapper so app.py does not have to change.
"""

from __future__ import annotations

import operator
from typing import Sequence, TypedDict
from typing_extensions import Annotated

from langgraph.graph import StateGraph
from langchain.tools import Tool

from legal_chains import (
    clause_reader,
    CaseLawRetrievalChain,
    risk_evaluator,
    improvement_chain,
)

# -----------------------------------------------------------
# 1. Shared state model
# -----------------------------------------------------------
class ClauseState(TypedDict, total=False):
    clause_id: str
    text: str
    clause_summary: str
    case_chunks: Annotated[Sequence[str], operator.add]   # fan-in reducer
    risk_assessment: str
    suggestion: str

# -----------------------------------------------------------
# 2. Graph nodes (each returns a partial state update)
# -----------------------------------------------------------
def read_clause(state: ClauseState) -> ClauseState:     # summarise clause
    summary = clause_reader.run(
        {"clause_id": state["clause_id"], "text": state["text"]}
    )
    return {"clause_summary": summary}

def retrieve_case_law(state: ClauseState) -> ClauseState:   # RAG
    qa = CaseLawRetrievalChain()
    excerpts = qa.run(state["clause_summary"])
    return {"case_chunks": [excerpts]}          # single-item list → concat OK

def evaluate_risk(state: ClauseState) -> ClauseState:       # score risks
    assessment = risk_evaluator.run(
        {
            "clause_summary": state["clause_summary"],
            "case_chunks": "\n\n".join(state.get("case_chunks", [])),
        }
    )
    return {"risk_assessment": assessment}

def suggest_improvement(state: ClauseState) -> ClauseState: # rewrite clause
    new_clause = improvement_chain.run(
        {
            "clause_summary": state["clause_summary"],
            "risk_assessment": state["risk_assessment"],
        }
    )
    return {"suggestion": new_clause}

# -----------------------------------------------------------
# 3. Build the graph
# -----------------------------------------------------------
def _build_graph():
    sg = StateGraph(ClauseState)

    sg.add_node("ReadClause", read_clause)
    sg.add_node("RetrieveCaseLaw", retrieve_case_law)
    sg.add_node("EvaluateRisk", evaluate_risk)
    sg.add_node("SuggestImprovement", suggest_improvement)

    sg.add_edge("ReadClause", "RetrieveCaseLaw")
    sg.add_edge("ReadClause", "EvaluateRisk")
    sg.add_edge("RetrieveCaseLaw", "EvaluateRisk")
    sg.add_edge("EvaluateRisk", "SuggestImprovement")

    sg.set_entry_point("ReadClause")
    sg.set_finish_point("SuggestImprovement")

    return sg.compile()            # returns a Runnable graph (invoke, stream…)

# -----------------------------------------------------------
# 4. Thin wrapper so app.py can still call .run()
# -----------------------------------------------------------
class _GraphRunner:
    """Gives the compiled graph a .run(input_dict) façade."""

    def __init__(self, graph):
        self._graph = graph

    def run(self, input_dict: ClauseState):
        # We only need the drafted clause for the UI; fall back to full dict
        out = self._graph.invoke(input_dict)    # <- current API :contentReference[oaicite:0]{index=0}
        return out.get("suggestion", out)       # keep old behaviour

def create_agent():
    """Entry point used by Streamlit."""
    graph = _build_graph()

    # (Tools list kept only if you need ReAct-style agent inspection/debug)
    tools = [
        Tool("ReadClause", read_clause, "Parse clause text."),
        Tool("RetrieveCaseLaw", retrieve_case_law, "Fetch relevant case law."),
        Tool("EvaluateRisk", evaluate_risk, "Assess legal risk."),
        Tool("SuggestImprovement", suggest_improvement, "Rewrite clause."),
    ]

    # Return wrapper so `agent.run({...})` still works in app.py
    return _GraphRunner(graph)
