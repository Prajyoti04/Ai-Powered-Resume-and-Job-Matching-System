import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# --- Patch for PyTorch + Python 3.13 ---
torch_device = "cpu"

# --- Safe model loading ---
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model._target_device = torch.device("cpu")  # prevent internal .to(device) calls
    return model

model = load_model()

# --- Sample job dataset ---
@st.cache_data
def load_data():
    data = {
        'Job Title': ['Data Scientist', 'Software Engineer', 'AI Researcher', 'Business Analyst', 'Machine Learning Engineer'],
        'Skills Required': [
            'Python, Machine Learning, Data Analysis, Deep Learning, SQL',
            'Java, C++, Software Development, Problem Solving, Git',
            'Artificial Intelligence, Research, Python, NLP, TensorFlow',
            'Business Analysis, Excel, Communication, SQL, Data Visualization',
            'Machine Learning, Python, TensorFlow, PyTorch, Data Engineering'
        ]
    }
    return pd.DataFrame(data)

jobs_df = load_data()

# --- Streamlit App ---
st.set_page_config(page_title="AI Resume Matcher", layout="centered")
st.title("🤖 AI-Powered Resume & Job Matching System")
st.write("Upload your resume and get the best-matching job suggestions based on your skills!")

resume_text = st.text_area("Paste your Resume Text here:", height=200)

if st.button("Find Matching Jobs"):
    if resume_text.strip():
        with st.spinner("Analyzing your resume..."):
            resume_embedding = model.encode(resume_text, convert_to_tensor=True, device=torch_device)
            job_embeddings = model.encode(jobs_df['Skills Required'].tolist(), convert_to_tensor=True, device=torch_device)

            similarities = util.pytorch_cos_sim(resume_embedding, job_embeddings)[0]
            jobs_df['Match Score (%)'] = [round(float(score) * 100, 2) for score in similarities]

            top_matches = jobs_df.sort_values(by='Match Score (%)', ascending=False).head(3)

        st.success("✅ Top Job Matches Found!")
        st.dataframe(top_matches[['Job Title', 'Match Score (%)']])
    else:
        st.warning("⚠️ Please enter your resume text first.")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Sentence Transformers")
