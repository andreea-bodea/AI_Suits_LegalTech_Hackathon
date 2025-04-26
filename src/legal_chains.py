from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ClauseReaderChain: parse clause
cla_arg = PromptTemplate(
    template="""
Clause ID: {clause_id}
Clause Text:
{text}

Summarize the clause obligations and parties.
""",
    input_variables=["clause_id", "text"]
)
clause_reader = LLMChain(
    llm=ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
    prompt=cla_arg
)

# CaseLawRetrievalChain: fetch case law via retriever
def CaseLawRetrievalChain(k=5, persist_dir="./case_law_index"):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vect = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vect.as_retriever(search_kwargs={"k": k})
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever
    )
    return qa

# RiskEvaluationChain: score risk
risk_prompt = PromptTemplate(
    template="""
Given this clause summary:
{clause_summary}
And these relevant case excerpts:
{case_chunks}

Identify and score the top legal risks (1–5) in bullet points.
""",
    input_variables=["clause_summary", "case_chunks"]
)
risk_evaluator = LLMChain(
    llm=ChatOpenAI(temperature=0),
    prompt=risk_prompt
)

# ImprovementSuggestionChain: rewrite clause
improve_prompt = PromptTemplate(
    template="""
Clause Summary:
{clause_summary}
Risk Assessment:
{risk_assessment}

Suggest an alternative clause wording to mitigate these risks.
""",
    input_variables=["clause_summary", "risk_assessment"]
)
improvement_chain = LLMChain(
    llm=ChatOpenAI(temperature=0),
    prompt=improve_prompt
)