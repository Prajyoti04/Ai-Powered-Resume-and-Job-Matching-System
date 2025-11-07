import os
import io
import base64
from dotenv import load_dotenv
import streamlit as st
import pdf2image
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import pdfplumber  # For extracting text from PDFs

# Load environment variables
load_dotenv()

# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize SentenceTransformer model on CPU to avoid device issues
try:
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
except NotImplementedError:
    st.error("Failed to load the SentenceTransformer model. This might be due to environment limitations. Please try again or contact support.")
    st.stop()

# Streamlit config
st.set_page_config(page_title="🤖 AI-Powered Resume & Job Matching System", layout="wide")

# Purple-pink theme
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
st.subheader("Upload your resume(s) and job description to get ATS-style evaluation!")

# Columns for input
col1, col2 = st.columns(2, gap="large")

with col1:
    input_text = st.text_area("📝 Job Description", height=300)
with col2:
    uploaded_files = st.file_uploader("📎 Upload Resume(s) (PDF only)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) Uploaded Successfully ✅")

# Convert PDF to base64 for Gemini
def input_pdf_setup(uploaded_file):
    images = pdf2image.convert_from_bytes(uploaded_file.read())
    first_page = images[0]
    img_byte_arr = io.BytesIO()
    first_page.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return [{"mime_type": "image/jpeg", "data": base64.b64encode(img_byte_arr).decode()}]

# Extract text from PDF for SentenceTransformer
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Get response from Gemini
def get_gemini_response(input_text, pdf_content, prompt):
    model_gen = genai.GenerativeModel('gemini-1.5-flash')
    response = model_gen.generate_content([input_text, pdf_content[0], prompt])
    return response.text

# Compute similarity using SentenceTransformer
def compute_similarity(job_desc, resume_text):
    job_embedding = model.encode(job_desc, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    similarity = util.cos_sim(job_embedding, resume_embedding).item()
    return round(similarity * 100, 2)  # Convert to percentage

# Buttons
submit1 = st.button("📄 Tell Me About the Resume")
submit2 = st.button("📊 Percentage Match")

input_prompt1 = """
You are an experienced Technical Human Resource Manager. 
Review the resume against the job description. Highlight strengths and weaknesses in relation to the role.
"""

# Responses
if submit1:
    if uploaded_files and input_text:
        for uploaded_file in uploaded_files:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_text, pdf_content, input_prompt1)
            st.subheader(f"Response for {uploaded_file.name}")
            st.write(response)
    else:
        st.warning("Please upload the resume(s) and provide a job description.")

if submit2:
    if uploaded_files and input_text:
        for uploaded_file in uploaded_files:
            resume_text = extract_text_from_pdf(uploaded_file)
            match_percentage = compute_similarity(input_text, resume_text)
            st.subheader(f"Percentage Match for {uploaded_file.name}")
            st.write(f"Match Percentage: {match_percentage}%")
            # Optionally, add more details like missing keywords using Gemini or simple keyword matching
            # For now, keep it simple with similarity score
    else:
        st.warning("Please upload the resume(s) and provide a job description.")

st.markdown('</div>', unsafe_allow_html=True)
