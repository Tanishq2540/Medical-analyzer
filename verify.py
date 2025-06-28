import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("medical_docs")
data = collection.get()
print(f"Total chunks in DB: {len(data['ids'])}")
