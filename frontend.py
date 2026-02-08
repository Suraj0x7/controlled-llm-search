import streamlit as st
from document_loader import load_document
from chunker import chunk_text
from embedder import create_embeddings
from vector_store import store_embeddings, search_embeddings
from llm_engine import LLMEngine

st.set_page_config(page_title="Controlled LLM + Semantic Search", layout="wide")

st.title("📄 Controlled LLM Inference with Semantic Search")

uploaded_file = st.file_uploader("Upload TXT or PDF", type=["pdf", "txt"])

if uploaded_file:
    with open(f"data/uploads/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.read())

    st.success("Document uploaded!")

    text = load_document(f"data/uploads/{uploaded_file.name}")
    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)
    store_embeddings(embeddings, chunks)

    st.info("Document processed and indexed!")

    query = st.text_input("Ask a question")

    col1, col2, col3 = st.columns(3)
    temperature = col1.slider("Temperature", 0.0, 1.0, 0.7)
    top_p = col2.slider("Top-P", 0.0, 1.0, 0.9)
    max_tokens = col3.slider("Max Tokens", 50, 500, 150)

    if st.button("Get Answer") and query:
        results = search_embeddings(query, top_k=3)
        context = "\n".join(results)

        llm = LLMEngine()

        prompt = f"""
Use only this context:

{context}

Question: {query}
Answer:
"""

        random_ans = llm.generate(prompt, max_tokens=max_tokens)

        deterministic_ans = llm.generate(prompt, max_tokens=max_tokens)

        st.subheader("🎲 Random Output")
        st.write(random_ans)

        st.subheader("📌 Deterministic Output")
        st.write(deterministic_ans)
