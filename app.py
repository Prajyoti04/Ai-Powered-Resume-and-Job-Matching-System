# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import os
import io
import base64
import streamlit as st
from PIL import Image
import pdf2image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import mysql.connector

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import pinecone

# Configure AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pinecone setup
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
index_name = "resume-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)
index = pinecone.Index(index_name)

# MySQL connection
def create_db_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQLHOST"),
            user=os.getenv("MYSQLUSER"),
            password=os.getenv("MYSQLPASSWORD"),
            database=os.getenv("MYSQLDATABASE"),
            port=int(os.getenv("MYSQLPORT", 3306))
        )
        print("MySQL connected!")
        return conn
    except mysql.connector.Error as err:
        print(f"MySQL connection error: {err}")
        return None

db_connection = create_db_connection()

# Helper functions
def input_pdf_setup(uploaded_file):
    images = pdf2image.convert_from_bytes(uploaded_file.read())
    first_page = images[0]
    img_byte_arr = io.BytesIO()
    first_page.save(img_byte_arr, format='JPEG')
    pdf_parts = [{"mime_type": "image/jpeg", "data": base64.b64encode(img_byte_arr.getvalue()).decode()}]
    return pdf_parts

def process_and_store_resumes(uploaded_files):
    for uploaded_file in uploaded_files:
        pdf_content = input_pdf_setup(uploaded_file)
        text = " ".join([part['data'] for part in pdf_content])
        vector = model.encode(text)
        index.upsert([(uploaded_file.name, vector.tolist())])

def get_gemini_response(input_text, pdf_content, prompt):
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
    response = model_ai.generate_content([input_text, pdf_content[0], prompt])
    return response.text

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="🤖 AI-Powered Resume & Job Matching System")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(to right, #9c27b0, #e91e63);
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .stTextArea>div>div>textarea {
            background-color: rgba(255,255,255,0.2);
            color: white;
        }
        .stFileUploader>div>div>div>input {
            background-color: rgba(255,255,255,0.2);
        }
        .css-1v3fvcr {
            padding: 1rem 1rem 1rem 1rem;
        }
    </style>
    """, unsafe_allow_html=True
)

st.title("🤖 AI-Powered Resume & Job Matching System")
st.markdown("Upload your resume(s) and job description to get an ATS-style evaluation and percentage match!")

# Input sections
st.markdown("### 📝 Job Description")
job_desc = st.text_area("Enter your Job Description here:")

st.markdown("### 📎 Upload Resume(s) (PDF only)")
uploaded_files = st.file_uploader("", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully!")
    process_and_store_resumes(uploaded_files)

# Buttons
col1, col2 = st.columns(2)

with col1:
    submit1 = st.button("Tell Me About the Resume")
with col2:
    submit2 = st.button("Percentage Match")

# Prompts
prompt_analysis = """
You are an experienced Technical Human Resource Manager. Review the resume against the job description and provide strengths, weaknesses, and alignment.
"""
prompt_percentage = """
You are a skilled ATS scanner. Evaluate the resume against the job description and provide percentage match, missing keywords, and final thoughts.
"""

# Actions
if submit1 and uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(job_desc, pdf_content, prompt_analysis)
        st.subheader(f"Analysis for {uploaded_file.name}")
        st.write(response)

if submit2 and uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(job_desc, pdf_content, prompt_percentage)
        st.subheader(f"Percentage Match for {uploaded_file.name}")
        st.write(response)
