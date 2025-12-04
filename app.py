import streamlit as st
from connect_memory import rag_chain  # uses the RAG chain you already built


# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="💊",
    layout="centered",
)

st.title("🩺 Medical PDF Chatbot")
st.caption("Answers strictly from your medical PDF knowledge base.")


# ----------------------------
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
# Chat Input
# ----------------------------
user_query = st.chat_input("Ask a question about the medical PDF...")

if user_query:
    # 1) Show user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query,
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    # 2) Run RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_query)
            answer = response["result"]
            source_docs = response.get("source_documents", [])

            # Show answer in UI
            st.markdown(answer)

            # Save assistant message + sources in history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": source_docs,
            })

            # Optional: show sources for this turn
            if source_docs:
                with st.expander("Show sources for this answer"):
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Source {i}:**")
                        st.write(doc.metadata)
                        st.write(doc.page_content[:400] + "...")
