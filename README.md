# PDF RAG Chatbot

A minimal Flask-based Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF, builds a FAISS vector store from the document text, and answers questions using a **local Ollama LLaMA model**.

---

## 🚀 Features

- PDF ingestion using `pdfplumber`
- Text chunking with `RecursiveCharacterTextSplitter`
- Embeddings via `sentence-transformers/all-MiniLM-L6-v2`
- Vector storage using **FAISS**
- Local LLM powered by **Ollama** (`llama3.2:3b`)
- Simple web UI: upload PDF → ask questions → get answers
- Fully offline (no external API calls)

---

## 🧰 Prerequisites

- Python **3.10+**
- Ollama installed and running locally  
- LLaMA model pulled:
  ```bash
  ollama pull llama3.2:3b
