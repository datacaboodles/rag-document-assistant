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

## Live Demo
[saivendingassistant.streamlit.app](https://saivendingassistant.streamlit.app)

# RAG Document Assistant — Sai Vending Services

An AI-powered document assistant that answers questions from 
operational documents using Retrieval-Augmented Generation (RAG).

## Live Demo
[saivendingassistant.streamlit.app](https://saivendingassistant.streamlit.app)

---

## How to Use the App

### Step 1 — Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Click **Sign Up** (use Google or email — free, no credit card needed)
3. After signing in, click **API Keys** in the left sidebar
4. Click **Create API Key**
5. Give it any name (example: "sai-vending-app")
6. Copy the key — it starts with `gsk_`
7. Keep it safe — you won't be able to see it again after closing

### Step 2 — Open the app

Go to [saivendingassistant.streamlit.app](https://saivendingassistant.streamlit.app)

### Step 3 — Paste your API key

Paste your Groq API key into the **"Enter your Groq API key"** box.
The key is never stored — it is only used for your current session.

### Step 4 — Ask a question
Type any question about the vending machine documents and press Enter.

---

## What Questions Can You Ask?

The assistant knows about 4 documents.
Here are example questions for each:

### Cleaning & Maintenance
- How often should I descale the machine?
- What are the daily cleaning steps?
- What cleaning tablets should I use?
- What does error code E12 mean?
- What does error code E05 mean?
- Coffee tastes bitter — what should I do?
- How do I clean the milk lines?
- What is the descaling solution SKU?

### Warranty
- What is covered under warranty?
- What is NOT covered by the warranty?
- How long is the warranty period?
- How do I make a warranty claim?
- Is water damage covered?
- How do I contact the service team?
- How quickly will a technician arrive?

### Customer Retention
- A customer hasn't ordered in 40 days — what should I do?
- How do I identify an at-risk customer?
- What discount should I offer a high-risk customer?
- When should I escalate a customer issue?
- What are upsell opportunities?
- How do I handle a medium-risk customer?

### Machine Installation
- What electrical socket does the machine need?
- How much clearance does the machine need?
- How heavy is the machine?
- What are the installation steps?
- Where is the serial number located?
- How many test cycles should I run before serving customers?
- What should I do within 7 days of installation?

---

## What the App Cannot Answer

- Questions outside these 4 documents
- Real-time information (stock levels, orders, live data)
- Personal account or customer-specific information
- Anything requiring internet search

If the answer is not in the documents, the assistant will say:
**"I don't have that information in the documents."**

---

## How It Works (Technical)
Your Question
↓
Converted to embedding vector (all-MiniLM-L6-v2)
↓
Compared against 4 document embeddings in FAISS
↓
Top 3 most relevant sections retrieved
↓
Sent to Groq LLM (Llama 3.3-70b) with context
↓
Answer generated and displayed

## Tech Stack
- Python
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS (Facebook AI Similarity Search)
- Groq API (Llama 3.3-70b-versatile)
- Streamlit

## Documents Included
- Cleaning SOP
- Warranty Policy
- Customer Retention Playbook
- Installation Guide

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Built For
Internal use at Sai Vending Services to allow field staff and 
account managers to get instant answers from operational documents 
without manual searching or calling the office.

