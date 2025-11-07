import streamlit as st
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="AI Resume Matcher", layout="centered")
st.title("🤖 AI-Powered Resume & Job Matching System")
st.write("Upload your resume or paste text to find best-matching jobs!")

# ------------------ Sample Job Data ------------------
@st.cache_data
def load_jobs():
    data = {
        "Job Title": [
            "Data Scientist",
            "Machine Learning Engineer",
            "Software Developer",
            "Business Analyst",
            "AI Researcher"
        ],
        "Skills Required": [
            "Python, Machine Learning, Deep Learning, SQL, Data Visualization",
            "TensorFlow, PyTorch, Python, MLOps, Model Optimization",
            "Java, C++, Git, Backend Development, REST APIs",
            "Excel, PowerBI, Data Analytics, Business Communication, SQL",
            "Artificial Intelligence, NLP, Python, Transformers, Research"
        ]
    }
    return pd.DataFrame(data)

jobs_df = load_jobs()

# ------------------ Resume Input ------------------
uploaded_file = st.file_uploader("📄 Upload your Resume (Text File Only):", type=["txt"])
resume_text = ""

if uploaded_file:
    resume_text = uploaded_file.read().decode("utf-8")
else:
    resume_text = st.text_area("Or paste your Resume text here:", height=200)

# ------------------ Matching Logic ------------------
if st.button("🔍 Find Matching Jobs"):
    if resume_text.strip():
        st.info("Analyzing resume using TF-IDF (lightweight engine)...")

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(jobs_df["Skills Required"].tolist() + [resume_text])
        cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

        jobs_df["Match Score (%)"] = [round(s * 100, 2) for s in cosine_sim]
        top_matches = jobs_df.sort_values(by="Match Score (%)", ascending=False).head(3)

        st.success("🎯 Top Job Matches Found!")
        st.dataframe(top_matches[["Job Title", "Match Score (%)"]])
    else:
        st.warning("⚠️ Please upload or enter your resume text first!")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit + Scikit-Learn")
