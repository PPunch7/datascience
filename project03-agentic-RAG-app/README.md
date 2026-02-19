# 🧠 Agentic RAG Application

An **Agentic Retrieval-Augmented Generation (RAG)** system built with:

- LangChain  
- CrewAI (Multi-Agent Orchestration)  
- FAISS (Vector Search)  
- Sentence-Transformers (Free Embeddings)  
- Tavily (Web Search)  
- Groq LLM  
- Streamlit (UI)  
- Docker (Containerized Deployment)  

---

## 🚀 Project Overview

This project implements a **multi-agent RAG architecture** that:

- Retrieves knowledge from a local PDF knowledge base
- Retrieves up-to-date information from the web
- Synthesizes findings using a Researcher agent
- Generates a structured report using a Writer agent
- Refines and improves the report using a Critic agent
- Displays results in a Streamlit web interface

---

## 🏗️ System Architecture
User Question
      ↓
Retriever (FAISS + SentenceTransformer)
      ↓
Web Search (Tavily)
      ↓
Context Combination
      ↓
Researcher Agent
      ↓
Writer Agent
      ↓
Critic Agent
      ↓
Final Structured Report

---

## 🧩 Multi-Agent Design

### 🔍 Researcher Agent
- Uses local RAG + web context
- Produces accurate summarized answers

### ✍ Writer Agent
- Converts research summary into structured professional report

### 🧐 Critic Agent
- Refines clarity, coherence, and correctness
- Produces final polished output

---

## 📂 Project Structure
project03-agentic-RAG-app/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── .env
│
└── app/
    ├── __init__.py
    ├── config.py
    ├── llm.py
    ├── vectorstore.py
    ├── tools.py
    ├── agents.py

---

## ⚙️ Installation (Local Development)

### 1️⃣ Create Virtual Environment

```bash
python -m venv rag_env
rag_env\Scripts\activate        # Windows
source rag_env/bin/activate   # Mac/Linux

### 2️⃣ Install Dependencies
`pip install -r requirements.txt`

### 3️⃣ Create .env File
`GROQ_API_KEY=your_groq_key`
`TAVILY_API_KEY=your_tavily_key`
`OPENAI_API_KEY=optional`

### ▶️ Run Locally (Streamlit)
`streamlit run app.py`

- Open in browser:
`http://localhost:8501`

## 🐳 Docker Deployment
### 1️⃣ Build Docker Image
`docker build -t agentic-rag-app .`

### 2️⃣ Run Container
`docker run --env-file .env -p 8501:8501 agentic-rag-app`


- Open:
`http://localhost:8501`

## 🔐 Environment Variables
- Variable	Description
- GROQ_API_KEY	LLM Provider
- TAVILY_API_KEY	Web Search API
- OPENAI_API_KEY	Optional (if using OpenAI embeddings)

## 🧠 RAG Pipeline
### Step 1 – Load PDF Knowledge Base
- PyPDFLoader
- Text splitting
- Embedding with SentenceTransformer

### Step 2 – Vector Store
- FAISS index built from embedded chunks

### Step 3 – Retrieval
- Top relevant documents retrieved

### Step 4 – Web Augmentation
- Tavily search integration

### Step 5 – Context Truncation
- Token-safe trimming (prevents LLM overflow)

### Step 6 – Multi-Agent Execution
- Researcher → Writer → Critic

##⚡ Performance Optimizations
- ✅ Local embeddings (no paid OpenAI)
- ✅ Context truncation (~3000 chars)
- ✅ Limited web results
- ✅ Controlled agent iterations
- ✅ Cached vectorstore in Streamlit
- ✅ Docker containerized runtime

## 🛠 Tech Stack
- Component	Technology
- LLM	Groq (Llama 3.x)
- Embeddings	Sentence-Transformers
- Vector DB	FAISS
- Orchestration	CrewAI
- Framework	LangChain
- UI	Streamlit
- Container	Docker

## 🎯 Example Questions
- What is generative AI in healthcare summarization?
- How does AI improve disease detection?
- What are risks of generative AI in healthcare?
- What is AI for health?