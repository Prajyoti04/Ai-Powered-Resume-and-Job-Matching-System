from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------ Helper Functions ------------------

def extract_resume_text(uploaded_file):
    """Extract text from uploaded PDF."""
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def get_gemini_response(job_description, resume_text, prompt):
    """Send resume text + job description to Gemini."""

    model = genai.GenerativeModel("gemini-1.5-flash")

    final_prompt = f"""
Job Description:
{job_description}

Resume:
{resume_text}

Instructions:
{prompt}
"""

    response = model.generate_content(final_prompt)

    return response.text


# ------------------ Streamlit UI ------------------

st.set_page_config(
    page_title="AI Resume & Job Matcher",
    page_icon="🤖"
)

st.title("🤖 AI-Powered Resume & Job Matching System")
st.write(
    "Upload your resume and job description to get an ATS-style evaluation and percentage match!"
)

col1, col2 = st.columns(2)

with col1:
    job_description = st.text_area(
        "📝 Job Description",
        key="input"
    )

with col2:
    uploaded_files = st.file_uploader(
        "📎 Upload Resume(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} Resume(s) Uploaded Successfully!")

prompt_eval = """
You are an experienced HR recruiter.

Review the resume based on the job description.

Highlight:

• Strengths

• Weaknesses

• Technical Skills

• Missing Skills

• Final Recommendation
"""

prompt_match = """
You are an ATS (Applicant Tracking System).

Compare the resume with the job description.

Return exactly in this format:

Percentage Match: xx%

Missing Keywords:

•

•

•

Final Evaluation:
"""


submit_eval = st.button("📄 Evaluate Resume")

submit_match = st.button("📊 Show Percentage Match")


if submit_eval:

    if uploaded_files:

        for uploaded_file in uploaded_files:

            resume_text = extract_resume_text(uploaded_file)

            response = get_gemini_response(
                job_description,
                resume_text,
                prompt_eval
            )

            st.subheader(f"🧠 Evaluation for {uploaded_file.name}")

            st.write(response)

    else:
        st.warning("Please upload a resume.")


elif submit_match:

    if uploaded_files:

        for uploaded_file in uploaded_files:

            resume_text = extract_resume_text(uploaded_file)

            response = get_gemini_response(
                job_description,
                resume_text,
                prompt_match
            )

            st.subheader(f"📊 Match Report for {uploaded_file.name}")

            st.write(response)

    else:
        st.warning("Please upload a resume.")
