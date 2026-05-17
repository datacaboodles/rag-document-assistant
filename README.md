# RAG Document Assistant — Sai Vending Services

An AI-powered document assistant that answers questions from 
operational documents using Retrieval-Augmented Generation (RAG).

## What it does
- Loads internal SOP and policy documents
- Splits them into chunks and creates semantic embeddings
- Stores embeddings in a FAISS vector index
- Retrieves the most relevant sections for any question
- Generates a grounded answer using Groq LLM (Llama 3.3)

## Tech stack
- Python
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS (Facebook AI Similarity Search)
- Groq API (Llama 3.3-70b)
- Streamlit

## Architecture
Documents → Chunks → Embeddings → FAISS Index
↑
User Query → Embedding → Similarity Search → Retrieved Chunks → Groq LLM → Answer

## Documents included
- Cleaning SOP
- Warranty Policy
- Customer Retention Playbook
- Installation Guide

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Built for
Internal use at Sai Vending Services to allow staff to get 
instant answers from operational documents without manual searching.
