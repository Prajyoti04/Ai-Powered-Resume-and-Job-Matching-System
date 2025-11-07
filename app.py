# app.py
import os
import io
import base64
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import pdf2image
import google.generativeai as genai
import pinecone
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------------
# Initialize Sentence Transformer
# ---------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------
# Initialize Pinecone (new SDK)
# ---------------------------
pinecone_client = pinecone.Client(
    api_key=os.getenv("PINECONE_API_KEY"), 
    environment="us-west1-gcp"
)

index_name = "resume-index"
if index_name not in pinecone_client.list_indexes():
    pinecone_client.create_index(index_name, dimension=384)

index = pinecone_client.Index(index_name)

# ---------------------------
# PDF processing
# ---------------------------
def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]

        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded or invalid PDF file.")

def process_and_store_resumes(uploaded_files):
    for uploaded_file in uploaded_files:
        pdf_content = input_pdf_setup(uploaded_file)
        text = " ".join([part['data'] for part in pdf_content])
        vector = model.encode(text)
        index.upsert([(uploaded_file.name, vector.tolist())])

# ---------------------------
# Gemini API function
# ---------------------------
def get_gemini_response(input_text, pdf_content, prompt):
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
    response = model_gemini.generate_content([input_text, pdf_content[0], prompt])
    return response.text

# ---------------------------
# Streamlit App UI
# ---------------------------
st.set_page_config(page_title="🤖 AI-Powered Resume & Job Matching", layout="wide")

# Custom CSS for purple/pink theme
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #d9a7c7, #fffcdc);
    font-family: 'Arial', sans-serif;
}
.stButton>button {
    background: linear-gradient(to right, #a18cd1, #fbc2eb);
    color: white;
    font-weight: bold;
    border-radius: 10px;
    height: 50px;
    width: 100%;
}
.stFileUploader>div {
    border: 2px dashed #a18cd1;
    border-radius: 10px;
    padding: 20px;
}
.stTextArea>div>textarea {
    border-radius: 10px;
    border: 2px solid #fbc2eb;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Powered Resume & Job Matching System")
st.markdown("Upload your resume and job description to get an ATS-style evaluation and percentage match!")

col1, col2 = st.columns(2, gap="large")

with col1:
    input_text = st.text_area("📝 Job Description:", key="input", height=250)
with col2:
    uploaded_files = st.file_uploader(
        "📎 Upload Resume(s) (PDF only)", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Drag and drop files here. Limit 200MB per file."
    )

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully!", icon="✅")
    with st.spinner("Processing resumes..."):
        process_and_store_resumes(uploaded_files)

# ---------------------------
# Buttons for evaluation
# ---------------------------
submit1 = st.button("Tell Me About the Resume")
submit3 = st.button("Percentage Match")

# Prompts
input_prompt1 = """
You are an experienced Technical Human Resource Manager. Review the provided resume against the job description. 
Highlight strengths, weaknesses, and alignment with the role.
"""

input_prompt3 = """
You are a skilled ATS scanner. Evaluate the resume against the job description and provide:
1. Match percentage
2. Keywords missing
3. Final thoughts
"""

# ---------------------------
# Button actions
# ---------------------------
if submit1:
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_text, pdf_content, input_prompt1)
            st.subheader(f"Response for {uploaded_file.name}:")
            st.write(response)
    else:
        st.warning("Please upload at least one resume.")

elif submit3:
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_text, pdf_content, input_prompt3)
            st.subheader(f"Response for {uploaded_file.name}:")
            st.write(response)
    else:
        st.warning("Please upload at least one resume.")
