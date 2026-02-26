from flask import Flask, request, jsonify
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route("/search", methods=["POST"])
def search():
    query = request.json.get("query", "")
    query_vector = model.encode([query])[0]
    
    # Compute similarity with all chunks
    similarities = [cosine_similarity(query_vector, model.encode([chunk])[0]) for chunk in chunks]
    
    # Get top 3
    top_idx = np.argsort(similarities)[-3:][::-1]
    results = [chunks[i] for i in top_idx]
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)