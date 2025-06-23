import streamlit as st
from dotenv import load_dotenv
import os

# Load API key and environment variables
load_dotenv()

# Import modular logic
from imageAnalysis import analyze_image
from text_qa import load_chroma_db, answer_question

# Load model, collection, and embedder only once
model, collection, embedder = load_chroma_db()

# Streamlit setup
st.set_page_config(page_title="VitalImage Analytics", page_icon="🧠")
st.title("MediBot: Medical Insight Assistant")
st.caption("Built to assist doctors and learners with AI-powered insights")

# Tabs
tab1, tab2 = st.tabs(["🖼️ Image Diagnosis", "💬 Medical Q&A"])

# --- Image Diagnosis Tab ---
with tab1:
    st.markdown("#### Upload a medical image (X-ray, MRI, etc.)")

    uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg", "avif"])
    if st.button("Generate Image Analysis"):
        if uploaded_file:
            result = analyze_image(uploaded_file, model)
            st.markdown("**Analysis Result:**")
            st.write(result)
        else:
            st.warning("Please upload an image.")

# --- Medical Q&A Tab ---
with tab2:
    st.markdown("#### Ask a medical question based on your reference book")
    user_question = st.text_input("Your question")
    
    if st.button("Get Answer"):
        if user_question:
            result = answer_question(user_question, model, collection, embedder)
            st.markdown("**Answer:**")
            st.write(result)
        else:
            st.warning("Please enter a question.")
