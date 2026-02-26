from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Create TF-IDF vectors for all chunks
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(chunks)

@app.route("/search", methods=["POST"])
def search():
    query = request.json.get("query", "")
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X).flatten()
    top_idx = np.argsort(sims)[-3:][::-1]
    results = [chunks[i] for i in top_idx]
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)