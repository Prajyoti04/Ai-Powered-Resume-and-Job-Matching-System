import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# Force CPU to avoid GPU-related NotImplementedError
device = "cpu"

# Load model safely
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device=device)

model = load_model()

# Streamlit UI
st.set_page_config(page_title="AI Resume-Job Matcher", page_icon="🤖", layout="wide")

st.title("🤖 AI-Powered Resume and Job Matching System")
st.write("Upload your resume and job description to see how well they match!")

# Input sections
st.header("📄 Resume Text")
resume_text = st.text_area("Paste your resume content here", height=250, placeholder="e.g., Experienced Data Scientist skilled in Python, ML, and NLP...")

st.header("💼 Job Description")
job_text = st.text_area("Paste the job description here", height=250, placeholder="e.g., Looking for an ML Engineer with 3+ years of experience in deep learning...")

# Matching Logic
if st.button("🔍 Match Now"):
    if resume_text.strip() == "" or job_text.strip() == "":
        st.warning("Please enter both resume and job description.")
    else:
        # Encode both texts
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        job_embedding = model.encode(job_text, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(resume_embedding, job_embedding)
        score = float(similarity.item()) * 100

        # Show results
        st.success(f"✅ Match Score: {score:.2f}%")
        if score > 80:
            st.info("🔥 Excellent match! Your resume fits this job very well.")
        elif score > 60:
            st.info("🙂 Good match! You meet many of the job requirements.")
        else:
            st.info("⚙️ Low match. Try updating your resume with relevant skills or keywords.")

# Footer
st.markdown("---")
st.markdown("💡 *Built with Streamlit + Sentence Transformers (MiniLM-L6-v2)*")
