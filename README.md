# 💼 Bank Terms & Services RAG Chatbot

This project is a **Retrieval-Augmented Generation (RAG) chatbot** built to answer questions about **Bank of America Online Banking Terms & Services**.  
It uses a PDF policy document as its knowledge base and provides accurate answers strictly based on the document.

---

## 📘 Project Goal

To build an AI-powered chatbot that can:
- Understand customer questions about online banking policies
- Retrieve accurate information from the official Bank Terms document
- Answer clearly using Groq's fast Llama 3.1 model
- Provide transparency by showing retrieved context

---

## 📄 Data Source

The knowledge base comes from this uploaded PDF:

**➡ `/mnt/data/Endreport.pdf`**  
(Loaded, parsed, chunked, and embedded into FAISS)

---

## 🧱 Tech Stack

| Component | Tool |
|----------|------|
| **LLM** | Groq — Llama 3.1 8B Instant |
| **Embeddings** | BAAI/bge-m3 (best free embedding model) |
| **Vector Store** | FAISS |
| **UI Framework** | Gradio |
| **PDF Parsing** | PyMuPDFLoader |
| **Text Chunking** | RecursiveCharacterTextSplitter |
| **Environment Management** | uv / virtualenv |
| **Backend Framework** | LangChain 0.2+ |

---

## 🚀 Features

### ✔ Retrieval-Augmented Generation (RAG)
The chatbot retrieves the most relevant passages from the document, then answers using the LLM.

### ✔ Full FAISS Integration
All PDF content is embedded with BGE-M3 and indexed for fast and accurate search.

### ✔ Groq-powered LLM
Uses **Llama-3.1-8B-Instant**, an extremely fast and capable model running on Groq LPU servers.

### ✔ Transparent Context
The UI shows:
- AI Answer  
- Retrieved Context used for generating the answer  

### ✔ Clean Gradio Interface
A simple, user-friendly dark UI:
- Input question field  
- Retrieved context panel  
- AI answer panel  

---


