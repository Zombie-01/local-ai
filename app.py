from flask import Flask, request, jsonify
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load index and chunks
index = faiss.read_index("book.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load small embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route("/search", methods=["POST"])
def search():
    query = request.json.get("query", "")
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector), k=3)  # top 3 chunks
    results = [chunks[i] for i in I[0]]
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)