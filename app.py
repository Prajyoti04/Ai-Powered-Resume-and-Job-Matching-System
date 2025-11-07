import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdf2image import convert_from_bytes
from PIL import Image
import io

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Resume & Job Matcher", layout="centered")
st.title("🤖 AI-Powered Resume & Job Matching System")
st.write("Upload your resume and get the best job suggestions using smart TF-IDF matching!")

# ------------------ SAMPLE JOB DATA ------------------
@st.cache_data
def load_jobs():
    data = {
        'Job Title': [
            'Data Scientist',
            'Software Engineer',
            'AI Researcher',
            'Business Analyst',
            'Machine Learning Engineer'
        ],
        'Skills Required': [
            'Python, Machine Learning, Data Analysis, Deep Learning, SQL',
            'Java, C++, Software Development, Problem Solving, Git',
            'Artificial Intelligence, Research, Python, NLP, TensorFlow',
            'Business Analysis, Excel, Communication, SQL, Data Visualization',
            'Machine Learning, Python, TensorFlow, PyTorch, Data Engineering'
        ]
    }
    return pd.DataFrame(data)

jobs_df = load_jobs()

# ------------------ RESUME INPUT ------------------
uploaded_file = st.file_uploader("📄 Upload your Resume (PDF/Text):", type=["pdf", "txt"])
resume_text = ""

if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        pdf_bytes = uploaded_file.read()
        images = convert_from_bytes(pdf_bytes)
        text = ""
        for image in images:
            text += image_to_text = ""  # placeholder if OCR added later
        st.info("✅ PDF uploaded successfully (text extraction simulated).")
        resume_text = "PDF resume uploaded. (You can add OCR extraction later.)"
    else:
        resume_text = uploaded_file.read().decode("utf-8")
else:
    resume_text = st.text_area("Or paste your Resume text here:", height=200)

# ------------------ MATCHING FUNCTION ------------------
if st.button("🔍 Find Matching Jobs"):
    if resume_text.strip():
        with st.spinner("Analyzing your resume..."):
            vectorizer = TfidfVectorizer(stop_words='english')
            job_desc = jobs_df['Skills Required'].tolist()
            tfidf_matrix = vectorizer.fit_transform(job_desc + [resume_text])
            
            cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
            jobs_df['Match Score (%)'] = [round(score * 100, 2) for score in cosine_sim]
            
            top_matches = jobs_df.sort_values(by='Match Score (%)', ascending=False).head(3)

        st.success("🎯 Top Job Matches Found!")
        st.dataframe(top_matches[['Job Title', 'Match Score (%)']])
    else:
        st.warning("⚠️ Please upload or enter your resume text first!")

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("Made with ❤️ using Streamlit & Scikit-Learn")
