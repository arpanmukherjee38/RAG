import json
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# --- 1. Configuration ---
PDF_PATH = "Retrieval.pdf"  # <-- Make sure to put your PDF here
EMBED_MODEL = "all-MiniLM-L6-v2" # A good, fast embedding model
CHUNK_SIZE = 800  # Characters per chunk
OVERLAP = 100     # Characters to overlap between chunks

print("=== Step 1: Reading and Chunking PDF ===")
try:
    reader = PdfReader(PDF_PATH)
    text = "".join(page.extract_text() for page in reader.pages)
    print(f"[SUCCESS] Read {len(reader.pages)} pages from {PDF_PATH}.")
except Exception as e:
    print(f"[ERROR] Could not read PDF. Make sure '{PDF_PATH}' exists. Error: {e}")
    exit()

# Split text into chunks
chunks = []
i = 0
while i < len(text):
    chunks.append(text[i:i + CHUNK_SIZE])
    i += CHUNK_SIZE - OVERLAP
print(f"Created {len(chunks)} text chunks.")


print("\n=== Step 2: Creating Text Embeddings ===")
print(f"Loading embedding model: '{EMBED_MODEL}'...")
try:
    embedder = SentenceTransformer(EMBED_MODEL)
    vecs = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    print(f"[SUCCESS] Created embeddings with shape: {vecs.shape}")
except Exception as e:
    print(f"[ERROR] Could not create embeddings. Error: {e}")
    exit()


print("\n=== Step 3: Building and Saving FAISS Index ===")
# Normalize vectors for accurate similarity search
faiss.normalize_L2(vecs)

# Build the FAISS index
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
print(f"FAISS index created with {index.ntotal} vectors.")

# Save the artifacts
faiss.write_index(index, "paper.index")
with open("paper.chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
with open("paper.meta.json", "w") as f:
    json.dump({"embed_model": EMBED_MODEL}, f, indent=2)

print("\n[COMPLETE] Project artifacts (paper.index, paper.chunks.json, paper.meta.json) are saved.")