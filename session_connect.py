# session_connect.py

import os
import numpy as np
from dotenv import load_dotenv

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFaceEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# -------------------------------------
# 1. Load HF Token
# -------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN missing in .env (add HF_TOKEN=...)")

# -------------------------------------
# 2. Choose Model (Zephyr or Mistral)
# -------------------------------------
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
# Or:
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# -------------------------------------
# 3. Create Chat LLM
# -------------------------------------
def load_llm():
    base_llm = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        huggingfacehub_api_token=HF_TOKEN,
        task="conversational",   # important for chat models
        max_new_tokens=512,
        temperature=0.4,
    )
    return base_llm

chat_llm = ChatHuggingFace(llm=load_llm())

# -------------------------------------
# 4. Prompt + LLM Chain
# -------------------------------------
SYSTEM_MSG = (
    "You are a helpful assistant that answers only from the given PDF. "
    "Use ONLY the provided context from the document. "
    "If the context is empty or not relevant to the question, "
    "respond EXACTLY with: I don't know."
)

HUMAN_TEMPLATE = """
Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("human", HUMAN_TEMPLATE),
    ]
)

llm_chain = prompt | chat_llm | StrOutputParser()

# -------------------------------------
# 5. Embeddings + Text Splitter (shared)
# -------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# -------------------------------------
# 6. Similarity Helpers
# -------------------------------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def build_context(question: str, docs, threshold: float = 0.55):
    """
    Given a question and a list of docs (chunks), compute cosine similarity
    and return (context_string, filtered_docs). If nothing is relevant,
    returns ("", []).
    """
    if not docs:
        return "", []

    query_emb = embedding_model.embed_query(question)
    scored = []

    for d in docs:
        doc_emb = embedding_model.embed_query(d.page_content)
        score = cosine_similarity(query_emb, doc_emb)
        scored.append((d, score))

    # Keep only relevant docs
    filtered = [d for (d, s) in scored if s >= threshold]

    if not filtered:
        return "", []

    context = "\n\n".join(d.page_content for d in filtered)
    return context, filtered

# -------------------------------------
# 7. Build FAISS index from ONE PDF
# -------------------------------------
def build_db_from_pdf(file_path: str) -> FAISS:
    """
    Build a FAISS vector store from a single PDF file.
    This is called whenever a new PDF is uploaded for a session.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    if not documents:
        raise ValueError("No pages found in PDF.")

    chunks = splitter.split_documents(documents)
    db = FAISS.from_documents(chunks, embedding_model)
    return db

# -------------------------------------
# 8. Build RAG chain bound to that DB
# -------------------------------------
def build_rag_chain(db: FAISS):
    """
    Given a FAISS db (for one PDF), create a RAG chain that:
      - retrieves from this db
      - applies similarity filtering
      - calls the LLM
      - returns {'result', 'source_documents'}
    """
    retriever = db.as_retriever(search_kwargs={"k": 4})

    def rag_logic(question: str):
        # Step 1: retrieve docs from this PDF
        docs = retriever.invoke(question)

        # Step 2: build context with similarity filtering
        context, filtered_docs = build_context(question, docs)

        # Step 3: no relevant context → "I don't know."
        if context.strip() == "":
            return {
                "result": "I don't know.",
                "source_documents": docs,
            }

        # Step 4: call LLM with context + question
        answer = llm_chain.invoke({"context": context, "question": question})

        return {
            "result": answer,
            "source_documents": filtered_docs or docs,
        }

    return RunnableLambda(rag_logic)

# -------------------------------------
# 9. Optional CLI test
# -------------------------------------
if __name__ == "__main__":
    pdf_path = input("Path to PDF: ")
    db = build_db_from_pdf(pdf_path)
    rag_chain = build_rag_chain(db)

    q = input("Write Query Here: ")
    resp = rag_chain.invoke(q)
    print("\nANSWER:\n", resp["result"])

    print("\nSOURCE DOCUMENTS:")
    for i, d in enumerate(resp["source_documents"], 1):
        print(f"\n--- Source {i} ---")
        print(d.metadata)
        print(d.page_content[:200], "...")
