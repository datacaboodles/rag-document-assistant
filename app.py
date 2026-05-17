
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from groq import Groq

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sai Vending Assistant",
    page_icon="☕",
    layout="centered"
)

# ── Load and chunk documents ──────────────────────────────────────────────────
def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r") as f:
                documents.append({"text": f.read(), "source": filename})
    return documents

def split_into_chunks(documents, chunk_size=100, overlap=20):
    all_chunks = []
    for doc in documents:
        words = doc["text"].split()
        start = 0
        chunk_num = 0
        while start < len(words):
            chunk_text = " ".join(words[start:start + chunk_size])
            all_chunks.append({
                "text": chunk_text,
                "source": doc["source"],
                "chunk_id": f"{doc['source']}_chunk_{chunk_num}"
            })
            chunk_num += 1
            start = start + chunk_size - overlap
    return all_chunks

# ── Build index — cached so it only runs once ─────────────────────────────────
@st.cache_resource
def build_index():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    documents = load_documents(".")
    chunks = split_into_chunks(documents)
    embeddings = np.array(
        model.encode([c["text"] for c in chunks])
    ).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index, chunks

# ── Search ────────────────────────────────────────────────────────────────────
def search(query, model, index, chunks, k=3):
    query_embedding = np.array(model.encode([query])).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [{
        "text": chunks[idx]["text"],
        "source": chunks[idx]["source"],
        "distance": distances[0][i]
    } for i, idx in enumerate(indices[0])]

# ── Ask LLM ───────────────────────────────────────────────────────────────────
def ask_llm(query, results, api_key):
    context = "\n\n".join([
        f"[From {r['source']}]:\n{r['text']}" for r in results
    ])
    prompt = f"""You are a helpful assistant for Sai Vending Services.
Answer using ONLY the context below. If the answer is not there, say so.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("☕ Sai Vending Assistant")
st.caption("Ask anything about cleaning, warranty, installation, or customer retention.")

groq_api_key = st.text_input(
    "Enter your Groq API key",
    type="password",
    placeholder="gsk_..."
)

query = st.text_input(
    "Your question",
    placeholder="How often should I descale the machine?"
)

if query and groq_api_key:
    with st.spinner("Searching documents..."):
        model, index, chunks = build_index()
        results = search(query, model, index, chunks, k=3)
        answer = ask_llm(query, results, groq_api_key)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("View retrieved document sections"):
        for i, r in enumerate(results):
            st.markdown(f"**Result {i+1} — {r['source']}**")
            st.write(r["text"])
            st.divider()

elif query and not groq_api_key:
    st.warning("Please enter your Groq API key above to get answers.")
