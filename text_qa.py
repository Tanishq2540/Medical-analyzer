# text_qa.py

import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load environment variables and configure Gemini
def load_chroma_db():
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    generation_config = {
        "temperature": 0.8,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    # ✅ New Chroma initialization (no legacy config!)
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("medical_docs")

    # Sentence embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return model, collection, embedder


# Given a user question, retrieve relevant text chunks and generate an answer
def answer_question(question, model, collection, embedder):
    embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=3)

    if not results["documents"][0]:
        return "I'm sorry, I couldn't find relevant information to answer that question."

    context = "\n\n".join(results["documents"][0])

    prompt = f"""You are a medical assistant. 
    Use only the following context to answer the user's question.
    Be precise and direct and give your answer in short bullet point paragraphs.
    Suggest symptoms and diagnosis also.
    If you don't know any answer, just output "Please provide a valid medical question.".

Context:
{context}

Question: {question}
"""
    response = model.generate_content(prompt)
    return response.text
