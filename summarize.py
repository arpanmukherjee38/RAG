import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# --- 1. Configuration ---
OLLAMA_MODEL = "gemma:2b"
# The question you want to ask the paper
QUERY = "What is the main conclusion of this paper?"

print("=== Step 1: Loading Knowledge Base and Embedder ===")
try:
    index = faiss.read_index("paper.index")
    with open("paper.chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open("paper.meta.json", "r") as f:
        meta = json.load(f)
    print("[SUCCESS] Loaded FAISS index, text chunks, and metadata.")
except Exception as e:
    print(f"[ERROR] Could not load project artifacts. Did you run 'build_index.py' first? Error: {e}")
    exit()

embedder = SentenceTransformer(meta["embed_model"])
print(f"Loaded embedder model: '{meta['embed_model']}'")

print("\n=== Step 2: Retrieving Relevant Context ===")
# Encode the user's query
query_vector = embedder.encode([QUERY], convert_to_numpy=True)
faiss.normalize_L2(query_vector)

# Search the FAISS index for the top 4 most relevant chunks
k = 4
distances, indices = index.search(query_vector, k)

# Get the actual text chunks
retrieved_chunks = [chunks[i] for i in indices[0]]
context = "\n\n".join(retrieved_chunks)
print(f"Retrieved {len(retrieved_chunks)} relevant chunks for the query.")


print("\n=== Step 3: Generating Summary with RAG ===")
# Construct the prompt for the LLM
prompt = f"""
Based ONLY on the following context from an academic paper, answer the user's question.
If the context doesn't contain the answer, state that you cannot answer based on the provided text.
Cite your answer by referring to the text.

--- CONTEXT ---
{context}
--- END OF CONTEXT ---

QUESTION: {QUERY}

ANSWER:
"""

print(f"Sending prompt to Ollama model: '{OLLAMA_MODEL}'...")
try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=60 # seconds
    )
    response.raise_for_status()
    summary = response.json().get("response", "").strip()

    print("\n--- GENERATED SUMMARY ---")
    print(summary)
    print("-------------------------")

except requests.exceptions.RequestException as e:
    print(f"\n[ERROR] Could not connect to Ollama. Is 'ollama serve' running? Error: {e}")