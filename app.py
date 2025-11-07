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
import numpy as np

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Force CPU to avoid Streamlit GPU errors
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Streamlit page config
st.set_page_config(page_title="🤖 AI-Powered Resume & Job Matching System", layout="wide")

# Custom CSS for purple-pink theme
st.markdown("""
    <style>
    .main {
        background: linear-gradient(120deg, #d5a6ff, #ffb6c1);
        border-radius: 15px;
        padding: 20px;
    }
    textarea, .stFileUploader {
        background: rgba(255,255,255,0.8);
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button {
        background: #9b59b6;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 1em;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🤖 AI-Powered Resume & Job Matching System")
st.subheader("Upload your resume(s) and job description to get an ATS-style evaluation!")

# Columns for inputs
col1, col2 = st.columns(2, gap="large")

with col1:
    input_text = st.text_area("📝 Job Description", height=300)
with col2:
    uploaded_files = st.file_uploader("📎 Upload Resume(s) (PDF only)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) Uploaded Successfully ✅")

def input_pdf_setup(uploaded_file):
    images = pdf2image.convert_from_bytes(uploaded_file.read())
    first_page = images[0]
    img_byte_arr = io.BytesIO()
    first_page.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return [
        {"mime_type": "image/jpeg", "data": base64.b64encode(img_byte_arr).decode()}
    ]

def get_gemini_response(input_text, pdf_content, prompt):
    model_gen = genai.GenerativeModel('gemini-1.5-flash')
    response = model_gen.generate_content([input_text, pdf_content[0], prompt])
    return response.text

# Buttons
submit1 = st.button("📄 Tell Me About the Resume")
submit2 = st.button("📊 Percentage Match")

input_prompt1 = """
You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt2 = """
You are a skilled ATS (Applicant Tracking System) scanner. Evaluate the resume against the job description. 
Provide percentage match first, then keywords missing, then final thoughts.
"""

if submit1:
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_text, pdf_content, input_prompt1)
            st.subheader(f"Response for {uploaded_file.name}")
            st.write(response)
    else:
        st.warning("Please upload the resume(s).")

if submit2:
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_text, pdf_content, input_prompt2)
            st.subheader(f"Percentage Match for {uploaded_file.name}")
            st.write(response)
    else:
        st.warning("Please upload the resume(s).")

st.markdown('</div>', unsafe_allow_html=True)
