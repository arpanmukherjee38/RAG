import streamlit as st
import json
import faiss
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import os

# --- App Configuration ---
st.set_page_config(
    page_title="Academic Paper Summarizer",
    page_icon="📚",
    layout="wide"
)

# --- Backend Logic (Functions from your scripts) ---

# This function builds the knowledge base. We use a Streamlit cache to avoid re-running it
# every time the app reloads, which saves a lot of time.
@st.cache_resource
def build_index(pdf_path, embed_model_name):
    """Reads a PDF, chunks it, creates embeddings, and builds a FAISS index."""
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

def generate_summary(query, index, chunks, embed_model, ollama_model):
    """Performs RAG to generate a summary."""
    try:
        embedder = SentenceTransformer(embed_model)
        query_vector = embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)

        k = 4  # Retrieve top 4 relevant chunks
        distances, indices = index.search(query_vector, k)
        retrieved_chunks = [chunks[i] for i in indices[0]]
        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
        Based ONLY on the following context from an academic paper, answer the user's question.
        If the context doesn't contain the answer, state that you cannot answer based on the provided text.
        The answer should be a concise summary.

        --- CONTEXT ---
        {context}
        --- END OF CONTEXT ---

        QUESTION: {query}

        ANSWER:
        """

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": ollama_model, "prompt": prompt, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to Ollama. Is 'ollama serve' running? Error: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred during generation: {e}")
        return None

# --- Frontend UI using Streamlit ---

st.title("📚 Academic Paper Summarizer")
st.markdown("Upload a PDF of an academic paper, and this app will use a local AI model to answer your questions about it.")

# --- UI Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    # Let the user choose which Ollama model to use
    ollama_model = st.selectbox(
        "Choose an Ollama Model",
        ("gemma:2b", "phi-3:mini", "llama3"),
        index=0  # Default to gemma:2b
    )
    st.info("Make sure you have pulled your chosen model in Ollama (e.g., `ollama pull gemma:2b`)")
    EMBED_MODEL = "all-MiniLM-L6-v2" # Embedding model is fixed for simplicity

# --- Main App Logic ---

# 1. File Uploader
uploaded_file = st.file_uploader("Upload your academic paper (PDF)", type="pdf")

# Use a session state to track if the index is built
if 'index_built' not in st.session_state:
    st.session_state.index_built = False

if uploaded_file is not None:
    # Save the uploaded file to a temporary path
    temp_dir = "temp_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. Build Index Button
    if st.button("Process Paper"):
        with st.spinner("Reading PDF and building knowledge base... This may take a moment."):
            # Call the backend function to build the index
            index, chunks = build_index(file_path, EMBED_MODEL)
            if index is not None and chunks is not None:
                # Save the results in the session state to use later
                st.session_state.faiss_index = index
                st.session_state.text_chunks = chunks
                st.session_state.index_built = True
                st.success("Paper processed! You can now ask questions about it.")
            else:
                st.session_state.index_built = False # Ensure state is reset on failure

    # 3. Query Input and Summarization
    if st.session_state.index_built:
        st.markdown("---")
        st.header("Ask a Question")
        query = st.text_area(
            "Enter your question or ask for a summary",
            "What is the main conclusion of this paper?"
        )

        if st.button("Generate Summary"):
            with st.spinner(f"Generating summary with '{ollama_model}'..."):
                # Retrieve the index and chunks from the session state
                index = st.session_state.faiss_index
                chunks = st.session_state.text_chunks
                # Call the RAG function to get the answer
                summary = generate_summary(query, index, chunks, EMBED_MODEL, ollama_model)
                if summary:
                    st.success("Summary Generated!")
                    st.markdown(summary)
else:
    st.session_state.index_built = False # Reset state if no file is uploaded