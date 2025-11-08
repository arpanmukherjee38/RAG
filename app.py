import streamlit as st
import json
import faiss
import numpy as np
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import google.generativeai as genai
genai.configure(api_key="GOOGLE_API_KEY")   # <-- put your google API here later

st.set_page_config(
    page_title="Academic Paper Summarizer",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def build_index(pdf_path, embed_model_name):
    try:
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() for page in reader.pages)
        if not text:
            st.error("Could not extract text from the PDF. The file might be empty or corrupted.")
            return None, None

        chunks = []
        chunk_size, overlap = 800, 100
        i = 0
        while i < len(text):
            chunks.append(text[i:i + chunk_size])
            i += chunk_size - overlap

        embedder = SentenceTransformer(embed_model_name)
        vecs = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(vecs)

        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)

        return index, chunks
    except Exception as e:
        st.error(f"An error occurred during indexing: {e}")
        return None, None

def generate_summary(query, index, chunks, embed_model):
    try:
        embedder = SentenceTransformer(embed_model)
        query_vector = embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)

        k = 4
        distances, indices = index.search(query_vector, k)
        retrieved_chunks = [chunks[i] for i in indices[0]]
        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
        Based ONLY on the following context from an academic paper, answer the user's question.
        If the context doesn't contain the answer, say you cannot answer.

        --- CONTEXT ---
        {context}
        --- END CONTEXT ---

        QUESTION: {query}

        ANSWER:
        """

        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        st.error(f"An error occurred during generation: {e}")
        return None

st.title("📚 Academic Paper Summarizer (Google Gemini)")
st.markdown("Upload a PDF and ask questions about it.")

with st.sidebar:
    st.header("AI Settings")
    EMBED_MODEL = "all-MiniLM-L6-v2"

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if 'index_built' not in st.session_state:
    st.session_state.index_built = False

if uploaded_file is not None:
    temp_dir = "temp_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Process Paper"):
        with st.spinner("Building knowledge base..."):
            index, chunks = build_index(file_path, EMBED_MODEL)
            if index is not None and chunks is not None:
                st.session_state.faiss_index = index
                st.session_state.text_chunks = chunks
                st.session_state.index_built = True
                st.success("Ready! Now ask a question.")
            else:
                st.session_state.index_built = False

    if st.session_state.index_built:
        st.markdown("---")
        st.header("Ask a Question")
        query = st.text_area("Ask something:", "What is the main conclusion?")

        if st.button("Generate Summary"):
            with st.spinner("Generating answer using Gemini..."):
                index = st.session_state.faiss_index
                chunks = st.session_state.text_chunks
                summary = generate_summary(query, index, chunks, EMBED_MODEL)
                if summary:
                    st.success("Answer:")
                    st.markdown(summary)
else:
    st.session_state.index_built = False
