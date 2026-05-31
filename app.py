import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

st.set_page_config(
    page_title="Sai Vending Assistant",
    page_icon="☕",
    layout="centered"
)

def load_documents_from_secrets():
    documents = []
    for name, content in st.secrets["documents"].items():
        documents.append({
            "text": content,
            "source": f"{name}.txt"
        })
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

@st.cache_resource
def build_index():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    documents = load_documents_from_secrets()
    chunks = split_into_chunks(documents)
    embeddings = np.array(
        model.encode([c["text"] for c in chunks])
    ).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index, chunks

def search(query, model, index, chunks, k=3):
    query_embedding = np.array(model.encode([query])).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [{
        "text": chunks[idx]["text"],
        "source": chunks[idx]["source"],
        "distance": distances[0][i]
    } for i, idx in enumerate(indices[0])]

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
