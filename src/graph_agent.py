from langgraph.graph.graph import Graph
from langchain.agents import Tool
from legal_chains import (
    clause_reader,
    CaseLawRetrievalChain,
    risk_evaluator,
    improvement_chain,
)

tools = [
    Tool(name="ReadClause", func=clause_reader.run,           description="Parse clause text."),
    Tool(name="RetrieveCaseLaw", func=CaseLawRetrievalChain().run, description="Fetch relevant case law."),
    Tool(name="EvaluateRisk", func=risk_evaluator.run,        description="Assess legal risk."),
    Tool(name="SuggestImprovement", func=improvement_chain.run, description="Rewrite clause to mitigate risk."),
]

def create_agent():
    llm = clause_reader.llm
    graph = Graph()

    # Add nodes (name, function) — no id= keyword!
    graph.add_node("ReadClause",        clause_reader.run)
    graph.add_node("RetrieveCaseLaw",   CaseLawRetrievalChain().run)
    graph.add_node("EvaluateRisk",      risk_evaluator.run)
    graph.add_node("SuggestImprovement", improvement_chain.run)

    # Wire edges (map outputs→inputs in state)
    graph.add_edge("ReadClause",      "RetrieveCaseLaw")
    graph.add_edge("ReadClause",      "EvaluateRisk")
    graph.add_edge("RetrieveCaseLaw", "EvaluateRisk")
    graph.add_edge("EvaluateRisk",    "SuggestImprovement")
    graph.add_edge("ReadClause",      "SuggestImprovement")

    # Define entry and exit
    graph.set_entry_point("ReadClause")
    graph.set_finish_point("SuggestImprovement")

    return graph.create_agent(llm=llm, tools=tools)