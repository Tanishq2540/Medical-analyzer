# 🩺 AI Medical Assistant (Gemini 1.5 Flash + RAG + ChromaDB + Streamlit)

A powerful medical assistant web app that uses **Retrieval-Augmented Generation (RAG)** to answer medical queries from a 4,500+ page textbook and analyze medical related images using **Gemini 1.5 Flash**.

---

## ✨ Features

- 🔍 **RAG-based Question Answering** from large-scale medical PDFs  
- 📚 Embeds 4,500+ pages into 6,700+ searchable chunks with SentenceTransformers  
- ⚡ Fast and relevant retrieval using **ChromaDB** vector store  
- 🖼️ **Medical Image Analysis** (e.g., X-rays, pathology) using Gemini 1.5 Flash  
- ✅ Clean, lightweight UI built with **Streamlit**  

---

## 🧠 How It Works

### 📖 Text-Based Q&A with RAG
1. **PDF Chunking**: Large medical textbook split into overlapping chunks.  
2. **Embedding**: Uses `all-MiniLM-L6-v2` via SentenceTransformers.  
3. **Vector Store**: Stores embeddings in ChromaDB.  
4. **Query**: User's question is embedded, top-k relevant chunks retrieved.  
5. **Answer Generation**: Gemini 1.5 Flash generates responses using retrieved context.  

### 🖼️ Medical Image Analysis
- Upload medical scans (e.g., X-ray, CT) and receive structured analysis.  
- Output includes:
  - Findings  
  - Possible implications  
  - Follow-up recommendations  
  - Confidence level  
- Powered by Gemini with strict medical prompting (non-diagnostic).  

---

## 🧾 Project Structure
```
├── app.py # Main Streamlit app
├── build_index.py # Builds vector DB from PDF
├── text_qa.py # Handles text-based QA using RAG
├── imageAnalysis.py # Gemini image analysis logic
├── requirements.txt
├── .env # Gemini API key (not tracked)
└── chroma_db/ # Local vector DB (excluded from Git)
```

🛑 Notes:  
`medical_book.pdf` and `chroma_db/` are `.gitignored` due to size limits.

---

## 🎥 Demo

[**📺 Watch Demo Video**](https://drive.google.com/file/d/1Bb04IEsVz2tqqIudtg8jywJurn3_-J99/view?usp=sharing)

