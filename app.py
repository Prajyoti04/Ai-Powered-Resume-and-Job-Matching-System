# app.py

from dotenv import load_dotenv
import os
import io
import base64
import streamlit as st
from PIL import Image
import pdf2image
import google.generativeai as genai
import pinecone
import numpy as np
from sentence_transformers import SentenceTransformer
import mysql.connector

# Load environment variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Sentence Transformer for vector embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
index_name = "resume-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)
index = pinecone.Index(index_name)

# MySQL connection setup (optional, remove if not needed)
def create_db_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQLHOST", "localhost"),
            user=os.getenv("MYSQLUSER", "root"),
            password=os.getenv("MYSQLPASSWORD", ""),
            database=os.getenv("MYSQLDATABASE", "codewthme")
        )
        print("MySQL Connection successfully created!")
        return conn
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        return None

db_connection = create_db_connection()

# Function to convert uploaded PDF to images
def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        pdf_parts = [{"mime_type": "image/jpeg", "data": base64.b64encode(img_byte_arr).decode()}]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded or invalid PDF file.")

# Function to process and store resumes in Pinecone
def process_and_store_resumes(uploaded_files):
    for uploaded_file in uploaded_files:
        pdf_content = input_pdf_setup(uploaded_file)
        text = " ".join([part['data'] for part in pdf_content])
        vector = model.encode(text)
        index.upsert([(uploaded_file.name, vector.tolist())])

# Gemini AI response function
def get_gemini_response(input_text, pdf_content, prompt):
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
    response = model_ai.generate_content([input_text, pdf_content[0], prompt])
    return response.text

# Streamlit UI configuration
st.set_page_config(page_title="ATS Resume Expert", layout="wide")
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #9b59b6, #e91e63);
    color: white;
    font-family: 'Arial', sans-serif;
}
.stButton>button {
    background-color: #8e44ad;
    color: white;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #e91e63;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🌸 ATS Resume Expert 🌸")
st.header("ATS Tracking System")

# Layout: two columns
col1, col2 = st.columns(2)
with col1:
    input_text = st.text_area("Job Description:", key="input")
with col2:
    uploaded_files = st.file_uploader(
        "Upload your resumes (PDF)...", type=["pdf"], accept_multiple_files=True
    )

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) Uploaded Successfully")
    with st.spinner("Processing resumes..."):
        process_and_store_resumes(uploaded_files)

# Buttons
submit1 = st.button("Tell Me About the Resume")
submit3 = st.button("Percentage Match")

# Input prompts
input_prompt1 = """
You are an experienced Technical Human Resource Manager. Review the resume against the job description.
Provide professional evaluation: strengths, weaknesses, and fit for the role.
"""
input_prompt3 = """
You are a skilled ATS scanner. Evaluate the resume against the job description.
Provide percentage match, missing keywords, and final thoughts.
"""

# Button actions
if submit1:
    if uploaded_files:
        pdf_content = input_pdf_setup(uploaded_files[0])
        response = get_gemini_response(input_text, pdf_content, input_prompt1)
        st.subheader("Resume Analysis:")
        st.write(response)
    else:
        st.warning("Please upload a resume first.")

elif submit3:
    if uploaded_files:
        pdf_content = input_pdf_setup(uploaded_files[0])
        response = get_gemini_response(input_text, pdf_content, input_prompt3)
        st.subheader("Resume Match Percentage:")
        st.write(response)
    else:
        st.warning("Please upload a resume first.")
