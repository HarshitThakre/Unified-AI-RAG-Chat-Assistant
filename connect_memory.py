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

# Recommended (chat model):
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

# Or you can switch to Mistral:
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"


# -------------------------------------
# 3. Create Base Conversational LLM
# -------------------------------------
def load_llm():
    base_llm = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        huggingfacehub_api_token=HF_TOKEN,
        task="conversational",       # model is chat-style
        max_new_tokens=512,
        temperature=0.4,
    )
    return base_llm


chat_llm = ChatHuggingFace(llm=load_llm())


# -------------------------------------
# 4. Prompt Template
# -------------------------------------
SYSTEM_MSG = (
    "You are a helpful medical assistant. "
    "Use ONLY the provided context from trusted medical PDFs. "
    "If the context is empty or not relevant to the question, respond EXACTLY with: I don't know."
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
# 5. Load FAISS Vector Store + Embeddings
# -------------------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True,
)

retriever = db.as_retriever(search_kwargs={"k": 4})  # get top 4 candidates


# -------------------------------------
# 6. Similarity Helpers (No hardcoding)
# -------------------------------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def build_context(question: str, docs, threshold: float = 0.55):
    """
    Compute embedding of question + each doc,
    filter by cosine similarity, and build final context string.
    Returns (context_string, filtered_docs).
    If no doc passes threshold → context_string is "" (forces 'I don't know').
    """
    if not docs:
        return "", []

    query_emb = embedding_model.embed_query(question)
    scored = []

    for d in docs:
        doc_emb = embedding_model.embed_query(d.page_content)
        score = cosine_similarity(query_emb, doc_emb)
        scored.append((d, score))

    # Filter by threshold
    filtered = [d for (d, s) in scored if s >= threshold]

    if not filtered:
        return "", []

    context = "\n\n".join(d.page_content for d in filtered)
    return context, filtered


# -------------------------------------
# 7. Core RAG Logic (single function)
# -------------------------------------
def rag_logic(question: str):
    """
    Full RAG pipeline:
      - retrieve candidate docs
      - compute similarity
      - if no good match -> 'I don't know.'
      - else -> send context + question to LLM
    Returns dict with 'result' and 'source_documents'.
    """
    # ✅ use retriever as a Runnable (no get_relevant_documents)
    docs = retriever.invoke(question)

    # Build context based on cosine similarity
    context, filtered_docs = build_context(question, docs)

    # If no relevant context → do NOT call LLM
    if context.strip() == "":
        answer = "I don't know."
        return {
            "result": answer,
            "source_documents": docs,
        }

    # Normal LLM flow with context
    answer = llm_chain.invoke({"context": context, "question": question})

    return {
        "result": answer,
        "source_documents": filtered_docs or docs,
    }


# Expose as a Runnable so you can use rag_chain.invoke(query)
rag_chain = RunnableLambda(rag_logic)


# -------------------------------------
# 8. CLI Test (optional)
# -------------------------------------
if __name__ == "__main__":
    q = input("Write Query Here: ")
    resp = rag_chain.invoke(q)
    print("\nANSWER:\n", resp["result"])

    print("\nSOURCE DOCUMENTS:")
    for i, d in enumerate(resp["source_documents"], 1):
        print(f"\n--- Source {i} ---")
        print(d.metadata)
        print(d.page_content[:200], "...")
