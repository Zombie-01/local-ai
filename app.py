from flask import Flask, request, render_template, jsonify
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

app = Flask(__name__)

# Load book chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Create TF-IDF vectorizer for retrieval
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(chunks)

# Function to find best sentence in a chunk
def best_sentence(chunk, query):
    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    if not sentences:
        return chunk
    s_vecs = vectorizer.transform(sentences)
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, s_vecs).flatten()
    best_idx = sims.argmax()
    return sentences[best_idx]

# Route: Web interface
@app.route("/")
def home():
    return '''
    <html>
    <body>
        <h2>Book Q&A (Offline)</h2>
        <form method="post" action="/ask">
            <input name="query" placeholder="Type your question..." size="50"/>
            <input type="submit" value="Ask"/>
        </form>
    </body>
    </html>
    '''

# Route: Process query
@app.route("/ask", methods=["POST"])
def ask():
    query = request.form.get("query", "")
    if not query:
        return "Query is empty"

    # Retrieve top 3 relevant chunks
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X).flatten()
    top_idx = sims.argsort()[-3:][::-1]

    # Concatenate retrieved chunks for LLM context
    context = "\n".join([chunks[i] for i in top_idx])

    # Prepare prompt for GPT4All
    prompt = f"Book excerpt:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Run GPT4All LLM via llama.cpp (CPU, GGML)
    cmd = ["./llama.cpp/build/main", "-m", "./ggml-gpt4all-j-v1.2-jazzy.bin", "-p", prompt, "-n", "200"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    answer = result.stdout.strip()
    return f"<h3>Q:</h3> {query} <br><h3>A:</h3> {answer}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)