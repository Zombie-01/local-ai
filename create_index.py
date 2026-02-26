from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import re

# -----------------------------
# 1️⃣ Load your book
# -----------------------------
book_path = r"D:\raspberry\book.txt"  # CHANGE if needed
with open(book_path, "r", encoding="utf-8") as f:
    text = f.read()

if len(text) < 10:
    raise ValueError("book.txt seems empty or too short!")

# -----------------------------
# 2️⃣ Split text into chunks
# -----------------------------
# Simple split by periods + basic cleaning
sentences = re.split(r'(?<=[.!?])\s+', text)
chunk_size = 5  # 5 sentences per chunk
chunks = []

for i in range(0, len(sentences), chunk_size):
    chunk = " ".join(sentences[i:i+chunk_size]).strip()
    if chunk:
        chunks.append(chunk)

print(f"Total chunks created: {len(chunks)}")

# -----------------------------
# 3️⃣ Create embeddings
# -----------------------------
print("Creating embeddings (may take a few minutes)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# -----------------------------
# 4️⃣ Build FAISS index
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# -----------------------------
# 5️⃣ Save index and chunks
# -----------------------------
faiss.write_index(index, "book.index")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Done! Saved 'book.index' and 'chunks.pkl'")