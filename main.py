import json
import gzip
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from os import getenv
from dotenv import load_dotenv

# ---------- EMBEDDING MODEL ----------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_embedding(text):
    return model.encode(text).tolist()

# ---------- LOAD DATA ONCE ----------
with gzip.open("movies.json.gz", "rt", encoding="utf-8") as f:
    DATA = json.load(f)

ROWS = []
EMBEDDINGS = []

for doc in DATA:
    if doc.get("embedding_hf") and doc.get("title"):
        ROWS.append(doc)
        EMBEDDINGS.append(doc["embedding_hf"])

if not EMBEDDINGS:
    raise RuntimeError("No embeddings found")

X = np.array(EMBEDDINGS)

# ---------- SEARCH ----------
def main_code(query):
    q = np.array(generate_embedding(query)).reshape(1, -1)
    scores = cosine_similarity(q, X)[0]

    top_k = scores.argsort()[-12:][::-1]

    result = []
    for i in top_k:
        poster = ROWS[i].get("poster")
        if not poster or not str(poster).startswith("http"):
            poster = None

        result.append({
            "title": ROWS[i]["title"],
            "poster": poster
        })

    return result

# ---------- LLM QUERY CLEANER ----------
load_dotenv()
GROQ_KEY = getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=GROQ_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.2
)

def filter_query(user_query):
    prompt = f"""
You are a query-refinement assistant.

Rules:
- Do not add new concepts
- Do not guess movie names
- Do not hallucinate
- Keep it short
- Output only the refined query

User input:
"{user_query}"

Refined query:
"""
    result = llm.invoke(prompt)
    return result.content.strip().strip('"').strip("'")

# ---------- FASTAPI ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/movie-result")
def movie_result(movie_des: str):
    refined = filter_query(movie_des)
    return main_code(refined)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
