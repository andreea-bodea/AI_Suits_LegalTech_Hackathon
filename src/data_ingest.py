import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load and chunk documents (rental clauses + case law)
def ingest_docs(input_dir: str, persist_dir: str, chunk_size=1000, chunk_overlap=200):
    print(f"Loading documents from: {input_dir}")
    loader = DirectoryLoader(input_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    if not docs:
        raise ValueError("No documents found. Please check the input directory and file types.")

    print(f"Loaded {len(docs)} documents.")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # embed and store
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print("Documents embedded and stored successfully.")
    return vectorstore

if __name__ == "__main__":
    ingest_docs(input_dir="/Users/andreeabodea/ANDREEA/LegalTechHackathon/AI_Suits_LegalTech_Hackathon/data", persist_dir="./case_law_index")