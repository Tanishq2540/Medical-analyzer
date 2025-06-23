import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Step 1: Initialize persistent ChromaDB client ---
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("medical_docs")

# --- Step 2: PDF chunking function ---
def extract_chunks(path, chunk_size=5000, overlap=350):
    doc = fitz.open(path)
    text = "".join([page.get_text() for page in doc])  # All pages
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# --- Step 3: Load and chunk the full PDF ---
chunks = extract_chunks("medical_book.pdf", chunk_size=3000, overlap=300)

# --- Step 4: Embed and store in Chroma ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

for i, chunk in enumerate(tqdm(chunks, desc="Embedding Chunks")):
    vector = embedder.encode(chunk).tolist()
    collection.add(documents=[chunk], embeddings=[vector], ids=[f"chunk_{i}"])

# --- Step 5: Optional (for clarity; handled automatically by PersistentClient) ---
client.persist()

print("✅ All pages indexed and saved to ./chroma_db")
