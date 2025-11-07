import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import os
from dotenv import load_dotenv
import pdfplumber
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------
# 🎯 Environment Setup
# ---------------------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="🤖 AI-Powered Resume & Job Match", layout="wide")

# ---------------------------------------------------------------------
# 🎯 Safe Model Loading (Fix for Torch 3.13 / Streamlit Cloud)
# ---------------------------------------------------------------------
try:
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
except NotImplementedError:
    import logging
    logging.warning("⚠️ Torch device conversion not supported. Loading model on CPU manually.")
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    st.error("❌ Error loading SentenceTransformer model.")
    st.write(e)
    st.stop()

# ---------------------------------------------------------------------
# 📄 Helper: Extract Text from Uploaded PDFs
# ---------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception:
        st.error("⚠️ Could not extract text from this PDF.")
    return text

# ---------------------------------------------------------------------
# 🧠 Helper: Compute Similarity Between Resume and Job Description
# ---------------------------------------------------------------------
def calculate_similarity(resume_text, job_text):
    if not resume_text.strip() or not job_text.strip():
        return 0.0
    try:
        resume_emb = model.encode(resume_text, convert_to_tensor=True)
        job_emb = model.encode(job_text, convert_to_tensor=True)
        score = torch.nn.functional.cosine_similarity(resume_emb, job_emb, dim=0)
        return float(score.item())
    except Exception as e:
        st.error("⚠️ Error calculating similarity:")
        st.write(e)
        return 0.0

# ---------------------------------------------------------------------
# 🎨 Streamlit UI
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2em;
            color: #9C27B0;
            text-align: center;
            font-weight: 700;
            margin-bottom: 1em;
        }
        .section-header {
            color: #E91E63;
            font-weight: 600;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            font-size: 1.2em;
        }
        .stButton>button {
            background-color: #9C27B0;
            color: white;
            font-weight: 600;
            border-radius: 10px;
            padding: 0.5em 1em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🤖 AI-Powered Resume & Job Matching System</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 📝 Job Description Input
# ---------------------------------------------------------------------
st.markdown('<div class="section-header">📝 Enter Job Description:</div>', unsafe_allow_html=True)
job_description = st.text_area("Paste the job description here", height=180)

# ---------------------------------------------------------------------
# 📎 Upload Resume(s)
# ---------------------------------------------------------------------
st.markdown('<div class="section-header">📎 Upload Resume (PDF only)</div>', unsafe_allow_html=True)
uploaded_resumes = st.file_uploader(
    "Upload one or more resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------------------------------------------------------------
# 🚀 Matching Logic
# ---------------------------------------------------------------------
if st.button("🔍 Match Resume(s)"):
    if not job_description.strip():
        st.warning("⚠️ Please enter a job description first.")
    elif not uploaded_resumes:
        st.warning("⚠️ Please upload at least one resume.")
    else:
        st.info("⏳ Analyzing... please wait.")
        results = []

        for resume_file in uploaded_resumes:
            resume_text = extract_text_from_pdf(resume_file)
            score = calculate_similarity(resume_text, job_description)
            results.append((resume_file.name, round(score * 100, 2)))

        st.success("✅ Analysis Complete!")

        # Display results
        for name, match in results:
            st.markdown(f"**📄 {name}** — Match Score: **{match}%**")
            st.progress(int(match))

# ---------------------------------------------------------------------
# 📢 Footer
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with 💜 using Streamlit & SentenceTransformers</p>",
    unsafe_allow_html=True
)
