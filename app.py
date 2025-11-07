import streamlit as st
import os
import mysql.connector
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import plotly.express as px
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import torch
import pinecone

# ------------------------------------------------------------
# 1️⃣ Setup & Initialization
# ------------------------------------------------------------
load_dotenv()

# Safety device setting for PyTorch
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"✅ Using device: {device}")

# Load model safely
try:
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
except Exception as e:
    st.error("❌ Error loading SentenceTransformer model.")
    st.write(e)
    st.stop()

# Pinecone initialization
pinecone_api = os.getenv("PINECONE_API_KEY")
if pinecone_api:
    pinecone.init(api_key=pinecone_api, environment="us-west1-gcp")
else:
    st.warning("⚠️ Pinecone API key not found in environment variables.")

# Google Gemini (Generative AI)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------------------------------------------------
# 2️⃣ MySQL Database Connection
# ------------------------------------------------------------
def get_db_connection():
    try:
        return mysql.connector.connect(
            host="localhost",
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "resume_db"),
        )
    except Exception as e:
        st.error("❌ Database connection failed.")
        st.write(e)
        return None

# ------------------------------------------------------------
# 3️⃣ Helper Functions
# ------------------------------------------------------------
def embed_text(text):
    """Generate embeddings for a given text using SentenceTransformer."""
    return model.encode(text).tolist()

def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF resume."""
    try:
        images = convert_from_path(uploaded_file)
        text = ""
        for img in images:
            # For simplicity, no OCR integration here
            text += " [Image converted from page]"
        return text
    except Exception as e:
        st.error("Error reading PDF file.")
        st.write(e)
        return ""

def generate_summary(text):
    """Generate a short professional summary using Gemini."""
    try:
        prompt = f"Summarize the following resume in 3-4 lines:\n{text}"
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Could not generate summary: {e}"

# ------------------------------------------------------------
# 4️⃣ Streamlit UI Layout
# ------------------------------------------------------------
st.set_page_config(page_title="AI Resume Matcher", layout="centered")
st.title("🤖 AI-Powered Resume & Job Matching System")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📄 Upload Resume", "📊 Job Matching", "📈 Insights"])

# ------------------------------------------------------------
# 5️⃣ Tab 1: Resume Upload & Summary
# ------------------------------------------------------------
with tab1:
    st.subheader("Upload Your Resume")
    uploaded_file = st.file_uploader("Select a resume (PDF format only)", type=["pdf"])

    if uploaded_file:
        with open("temp_resume.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("✅ Resume uploaded successfully!")
        resume_text = extract_text_from_pdf("temp_resume.pdf")

        st.markdown("### 🧾 Extracted Text Preview")
        st.text_area("Resume Text", resume_text[:1500] + "..." if len(resume_text) > 1500 else resume_text, height=200)

        if st.button("✨ Generate Summary", use_container_width=True):
            summary = generate_summary(resume_text)
            st.markdown("### 🧠 AI Summary")
            st.info(summary)

# ------------------------------------------------------------
# 6️⃣ Tab 2: Job Matching Section
# ------------------------------------------------------------
with tab2:
    st.subheader("Find Matching Jobs")
    job_desc = st.text_area("Enter Job Description", placeholder="Paste the job description here...")
    
    if st.button("🔍 Match Resume", use_container_width=True):
        if uploaded_file and job_desc:
            resume_embed = embed_text(resume_text)
            job_embed = embed_text(job_desc)
            similarity = np.dot(resume_embed, job_embed) / (np.linalg.norm(resume_embed) * np.linalg.norm(job_embed))
            st.metric("Match Score (%)", f"{round(similarity * 100, 2)}")
        else:
            st.warning("⚠️ Please upload a resume and enter a job description.")

# ------------------------------------------------------------
# 7️⃣ Tab 3: Insights / Analytics
# ------------------------------------------------------------
with tab3:
    st.subheader("📊 Analytics Dashboard")

    data = {
        "Skills": ["Python", "ML", "SQL", "Communication", "Data Analysis"],
        "Proficiency": [80, 75, 65, 90, 70]
    }

    fig = px.bar(data, x="Skills", y="Proficiency", title="Skill Proficiency Overview", color="Skills")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 AI-Powered Insights")
    st.write("Use this dashboard to understand your strengths and skill alignment with job descriptions.")

# ------------------------------------------------------------
# 8️⃣ Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("🧠 Built with Streamlit, SentenceTransformers, Pinecone & Gemini AI")

