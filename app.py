from dotenv import load_dotenv
import os
import io
import base64
import streamlit as st
import pdf2image
import google.generativeai as genai
from PIL import Image
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------ Helper Functions ------------------

def get_gemini_response(input_text, pdf_content, prompt):
    """Send resume + job description to Gemini for analysis"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, pdf_content[0], prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    """Convert uploaded PDF to image for Gemini input"""
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
        raise FileNotFoundError("No file uploaded.")

# ------------------ Streamlit App ------------------

st.set_page_config(page_title="AI Resume & Job Matcher", page_icon="🤖")
st.title("🤖 AI-Powered Resume & Job Matching System")
st.write("Upload your resume and job description to get an ATS-style evaluation and percentage match!")

# UI Layout
col1, col2 = st.columns(2)
with col1:
    job_description = st.text_area("📝 Job Description:", key="input")
with col2:
    uploaded_files = st.file_uploader("📎 Upload Resume(s) (PDF only)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} Resume(s) Uploaded Successfully!")

# Prompts for Gemini
prompt_eval = """
You are an experienced HR recruiter. Review the resume based on the job description. 
Highlight the candidate’s strengths, weaknesses, and how well they fit the role.
"""

prompt_match = """
You are an ATS (Applicant Tracking System). Compare the resume with the job description.
Output:
1️⃣ Percentage match
2️⃣ Missing keywords
3️⃣ Final evaluation summary.
"""

# Action Buttons
submit_eval = st.button("📄 Evaluate Resume")
submit_match = st.button("📊 Show Percentage Match")

# ------------------ Actions ------------------

if submit_eval:
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(job_description, pdf_content, prompt_eval)
            st.subheader(f"🧠 Evaluation for {uploaded_file.name}")
            st.write(response)
    else:
        st.warning("⚠️ Please upload at least one resume.")

elif submit_match:
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(job_description, pdf_content, prompt_match)
            st.subheader(f"📊 Match Report for {uploaded_file.name}")
            st.write(response)
    else:
        st.warning("⚠️ Please upload at least one resume.")
