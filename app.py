# app.py

import os
import io
import base64
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import pdf2image
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import pinecone
import numpy as np
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Sentence Transformer model (force CPU to avoid Streamlit errors)
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
index_name = "resume-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)
index = pinecone.Index(index_name)

# Function: convert PDF to base64 images for Gemini
def input_pdf_setup(uploaded_file):
    if uploaded_file:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        pdf_parts = [{"mime_type": "image/jpeg", "data": base64.b64encode(img_byte_arr).decode()}]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded or invalid PDF file.")

# Function: process resumes and store embeddings in Pinecone
def process_and_store_resumes(uploaded_files):
    for uploaded_file in uploaded_files:
        pdf_content = input_pdf_setup(uploaded_file)
        text = " ".join([part['data'] for part in pdf_content])
        vector = model.encode(text)
        index.upsert([(uploaded_file.name, vector.tolist())])

# Function: get response from Gemini
def get_gemini_response(input_text, pdf_content, prompt):
    model_gen = genai.GenerativeModel('gemini-1.5-flash')
    response = model_gen.generate_content([input_text, pdf_content[0], prompt])
    return response.text

# --- Streamlit UI ---
st.set_page_config(page_title="🤖 AI-Powered Resume & Job Matching System")
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #a18cd1, #fbc2eb);
    font-family: 'Arial', sans-serif;
}
.stButton>button {
    background-color: #8e44ad;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 200px;
}
.stTextArea>div>textarea {
    border-radius: 10px;
}
.stFileUploader>div {
    border-radius: 10px;
    background-color: rgba(255,255,255,0.2);
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Powered Resume & Job Matching System")
st.subheader("Upload your resume and job description to get an ATS-style evaluation and percentage match!")

# Layout: two columns
col1, col2 = st.columns(2, gap="large")

with col1:
    input_text = st.text_area("📝 Job Description", height=250)

with col2:
    uploaded_files = st.file_uploader("📎 Upload Resume(s) (PDF only)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) Uploaded Successfully")
    with st.spinner("Processing resumes..."):
        process_and_store_resumes(uploaded_files)

# Buttons for evaluation
submit1 = st.button("Tell Me About the Resume")
submit2 = st.button("Percentage Match")

# Prompts
prompt_evaluation = """
You are an experienced Technical HR Manager. Review the provided resume against the job description.
Highlight strengths and weaknesses and indicate whether the candidate's profile aligns with the role.
"""

prompt_percentage = """
You are a skilled ATS scanner. Evaluate the resume against the job description and provide:
1. Percentage match
2. Keywords missing
3. Final thoughts
"""

if submit1:
    if uploaded_files:
        for file in uploaded_files:
            pdf_content = input_pdf_setup(file)
            response = get_gemini_response(input_text, pdf_content, prompt_evaluation)
            st.subheader(f"Response for {file.name}:")
            st.write(response)
    else:
        st.warning("Please upload at least one resume.")

if submit2:
    if uploaded_files:
        for file in uploaded_files:
            pdf_content = input_pdf_setup(file)
            response = get_gemini_response(input_text, pdf_content, prompt_percentage)
            st.subheader(f"Match Analysis for {file.name}:")
            st.write(response)
    else:
        st.warning("Please upload at least one resume.")
