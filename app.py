import streamlit as st
import base64
import io
from PIL import Image
import pdf2image
import google.generativeai as genai
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pinecone
import numpy as np

# Load environment variables
load_dotenv()

# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
index_name = "resume-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)
index = pinecone.Index(index_name)

# ----------------------- Helper Functions -----------------------

def get_gemini_response(input, pdf_content, prompt):
    """Send resume and job description to AI model"""
    g_model = genai.GenerativeModel('gemini-1.5-flash')
    response = g_model.generate_content([input, pdf_content[0], prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    """Convert uploaded PDF to base64-encoded image"""
    if uploaded_file is not None:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        pdf_parts = [{
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_byte_arr).decode()
        }]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded or invalid PDF file.")

def process_and_store_resumes(uploaded_files):
    """Store PDF embeddings in Pinecone"""
    for uploaded_file in uploaded_files:
        pdf_content = input_pdf_setup(uploaded_file)
        text = " ".join([part['data'] for part in pdf_content])
        vector = model.encode(text)
        index.upsert([(uploaded_file.name, vector.tolist())])

# ----------------------- Streamlit UI -----------------------

st.set_page_config(page_title="🤖 ATS Resume Analyzer", layout="wide")

# Custom CSS for purple-pink theme
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #9b59b6, #ff79c6);
    font-family: 'Arial', sans-serif;
    color: #fff;
}
.stButton>button {
    background: #ff79c6;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.5em 1.2em;
}
.stFileUploader>div {
    background-color: rgba(255, 255, 255, 0.15);
    border-radius: 10px;
    padding: 1em;
    margin-bottom: 1em;
}
.stTextArea>div>div>textarea {
    border-radius: 10px;
    padding: 0.5em;
}
h1, h2, h3, h4 {
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align:center;'>🤖 AI-Powered Resume & Job Matching System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload your resume(s) and job description to get an ATS-style evaluation and percentage match!</p>", unsafe_allow_html=True)

# Input columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📝 Job Description")
    input_text = st.text_area("", placeholder="Paste your job description here...", height=250)

with col2:
    st.markdown("### 📎 Upload Resume(s) (PDF only)")
    uploaded_files = st.file_uploader("", type=["pdf"], accept_multiple_files=True)

# Show uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown(f"<div style='background-color: rgba(255, 255, 255, 0.15); border-radius: 10px; padding: 10px; margin-bottom:10px;'>✅ {uploaded_file.name} uploaded successfully!</div>", unsafe_allow_html=True)

# Buttons
st.markdown("<br>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    submit1 = st.button("Tell Me About the Resume")
with col4:
    submit3 = st.button("Percentage Match")

# Prompts
input_prompt1 = """
You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. Give me the percentage of match if the resume matches
the job description. First the output should come as percentage and then keywords missing and last final thoughts.
"""

# ----------------------- Button Actions -----------------------

if submit1:
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_text, pdf_content, input_prompt1)
            st.subheader(f"Response for {uploaded_file.name}")
            st.write(response)
    else:
        st.warning("Please upload at least one resume.")

elif submit3:
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_text, pdf_content, input_prompt3)
            st.subheader(f"Response for {uploaded_file.name}")
            st.write(response)
    else:
        st.warning("Please upload at least one resume.")
