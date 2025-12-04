# session_app.py

import os
import streamlit as st
from session_connect import build_db_from_pdf, build_rag_chain

st.set_page_config(
    page_title="Per-Session PDF Chat",
    page_icon="📄",
    layout="centered",
)


st.title("📄 PDF Chatbot (One PDF per session)")
st.caption("Upload a PDF, then ask questions strictly answered from that document.")


# Initialize Chat History
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {"role": "user"/"assistant", "content": str, "sources": [...]}


# ----------------------------
# Show Chat History
# ----------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources")
        if sources:
            with st.expander("Show sources"):
                for i, doc in enumerate(sources, 1):
                    st.markdown(f"**Source {i}:**")
                    st.write(doc.metadata)
                    st.write(doc.page_content[:400] + "...")

# ----------------------------
# Session State Initialization
# ----------------------------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

# ----------------------------
# Sidebar: Upload PDF
# ----------------------------
st.sidebar.header("📄 Upload PDF")

uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file is not None:
    # Save PDF to a temp folder
    upload_dir = "session_pdfs"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Build a fresh DB + RAG chain just for this PDF
    with st.spinner("Indexing PDF (building vector store)..."):
        db = build_db_from_pdf(file_path)
        rag_chain = build_rag_chain(db)

    # Reset session state for this new PDF
    st.session_state.rag_chain = rag_chain
    st.session_state.chat_history = []
    st.session_state.current_pdf = uploaded_file.name

    st.sidebar.success(f"✅ Loaded PDF: {uploaded_file.name}")
    st.sidebar.info("Now you can ask questions about this PDF.")

# Optional: New session button (clear everything)
if st.sidebar.button("🔁 Start New Session"):
    st.session_state.rag_chain = None
    st.session_state.chat_history = []
    st.session_state.current_pdf = None
    st.experimental_rerun()

# ----------------------------
# Show which PDF is active
# ----------------------------
if st.session_state.current_pdf:
    st.markdown(f"**Active PDF:** `{st.session_state.current_pdf}`")
else:
    st.warning("No PDF uploaded yet. Please upload a PDF from the sidebar.")

# ----------------------------
# Show Chat History
# ----------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources")
        if sources:
            with st.expander("Show sources"):
                for i, doc in enumerate(sources, 1):
                    st.markdown(f"**Source {i}:**")
                    st.write(doc.metadata)
                    st.write(doc.page_content[:400] + "...")

# ----------------------------
# Chat Input
# ----------------------------
user_query = st.chat_input("Ask a question about the active PDF...")

if user_query:
    if st.session_state.rag_chain is None:
        st.error("Please upload a PDF first.")
    else:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query,
        })

        with st.chat_message("user"):
            st.markdown(user_query)

        # Get answer from RAG
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(user_query)
                answer = response["result"]
                source_docs = response.get("source_documents", [])

                st.markdown(answer)

                # Save assistant response + sources
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,


                    "sources": source_docs,
                })

                if source_docs:
                    with st.expander("Show sources for this answer"):
                        for i, doc in enumerate(source_docs, 1):
                            st.markdown(f"**Source {i}:**")
                            st.write(doc.metadata)
                            st.write(doc.page_content[:400] + "...")
