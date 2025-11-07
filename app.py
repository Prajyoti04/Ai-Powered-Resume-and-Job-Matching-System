# app.py
from dotenv import load_dotenv
import os
import io
import base64
import streamlit as st
from PIL import Image
import pdf2image
import google.generativeai as genai
import plotly.graph_objects as go
import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone

# Load environment variables
load_dotenv()

# Configure AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
index_name = "resume-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)
index = pinecone.Index(index_name)

# ----------------- Utility Functions ----------------- #

def get_gemini_response(input_text, pdf_content, prompt):
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
    response = model_ai.generate_content([input_text, pdf_content[0], prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        pdf_parts = [{"mime_type": "image/jpeg", "data": base64.b64encode(img_byte_arr.getvalue()).decode()}]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded or invalid PDF file.")

def process_and_store_resumes(uploaded_files):
    for uploaded_file in uploaded_files:
        pdf_content = input_pdf_setup(uploaded_file)
        text = " ".join([part['data'] for part in pdf_content])
        vector = model.encode(text)
        index.upsert([(uploaded_file.name, vector.tolist())])

def show_response(title, response_text):
    st.markdown(
        f"""
        <div style='background-color: #e8eaf6; padding: 15px; border-radius: 10px; margin-top: 10px;'>
            <h4 style='color: #1a237e;'>{title}</h4>
            <p style='color: #333;'>{response_text}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def plot_match(match, keywords_found, keywords_missing):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Match %', 'Keywords Found', 'Keywords Missing'],
        y=[match, len(keywords_found), len(keywords_missing)],
        marker_color=['green', 'blue', 'red']
    ))
    fig.update_layout(
        title="Resume Evaluation Overview",
        yaxis_title="Count",
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------- Streamlit UI ----------------- #

st.set_page_config(page_title="ATS Resume Expert", page_icon="📄", layout="wide")

# Gradient background
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #f3e5f5, #e8eaf6);
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header section
st.markdown(
    """
    <div style='text-align: center; background-color: #f5f5f5; 
                padding: 20px; border-radius: 10px;'>
        <h1 style='color: #4B0082;'>ATS Resume Expert</h1>
        <p style='color: #333; font-size: 18px;'>
        Upload resumes and see how well they match the job description using AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Input columns
col1, col2 = st.columns([2, 1], gap="medium")

with col1:
    input_text = st.text_area(
        "Job Description",
        placeholder="Paste your job description here...",
        height=250
    )

with col2:
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

# Uploaded files feedback
if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully! ✅")
    for file in uploaded_files:
        st.write(f"• {file.name}")
    with st.spinner("Processing resumes..."):
        process_and_store_resumes(uploaded_files)

# Buttons
submit1 = st.button("Analyze Resume 📄")
submit3 = st.button("Percentage Match 📊")

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

# ----------------- Button Actions ----------------- #
if submit1:
    if uploaded_files:
        pdf_content = input_pdf_setup(uploaded_files[0])
        response = get_gemini_response(input_text, pdf_content, input_prompt1)
        show_response("Resume Analysis", response)
    else:
        st.warning("Please upload the resume")

elif submit3:
    if uploaded_files:
        pdf_content = input_pdf_setup(uploaded_files[0])
        response = get_gemini_response(input_text, pdf_content, input_prompt3)
        show_response("Percentage Match Analysis", response)
    else:
        st.warning("Please upload the resume")
