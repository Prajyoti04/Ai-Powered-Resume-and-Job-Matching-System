from dotenv import load_dotenv
import streamlit as st
import os
import io
import base64
from PIL import Image
import pdf2image
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------ Helper Functions ------------------ #
def input_pdf_setup(uploaded_file):
    """Convert uploaded PDF to base64 images for processing"""
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

def get_gemini_response(input_text, pdf_content, prompt):
    """Call Google Gemini model to generate response"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, pdf_content[0], prompt])
    return response.text

# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="🤖 AI Resume & Job Matching", layout="wide")

st.title("🤖 AI-Powered Resume & Job Matching System")
st.write("Upload your resume and job description to get ATS-style insights!")

# Layout: Job description & Resume upload
col1, col2 = st.columns(2)

with col1:
    input_text = st.text_area(
        "📝 Job Description", 
        height=250,
        placeholder="Paste the job description here..."
    )

with col2:
    uploaded_files = st.file_uploader(
        "📎 Upload Resume(s) (PDF only)", 
        type=["pdf"], 
        accept_multiple_files=True
    )

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) Uploaded Successfully", icon="✅")

# ------------------ Prompts ------------------ #
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

# ------------------ Buttons ------------------ #
st.write("")  # Spacer
submit1 = st.button("📄 Tell Me About the Resume")
st.write("")  # Spacer
submit3 = st.button("📊 Percentage Match")
st.write("")  # Spacer

# ------------------ Button Actions ------------------ #
if submit1:
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_text, pdf_content, input_prompt1)
            st.subheader(f"Resume: {uploaded_file.name}")
            st.write(response)
            st.write("---")  # Separator
    else:
        st.warning("Please upload at least one resume.")

elif submit3:
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_text, pdf_content, input_prompt3)
            st.subheader(f"Resume: {uploaded_file.name}")
            st.write(response)
            st.write("---")  # Separator
    else:
        st.warning("Please upload at least one resume.")
