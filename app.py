from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from updated_resume import (
    process_and_store_resumes,
    evaluate_resume,
    ats_match,
)

st.set_page_config(page_title="AI Resume & Job Matcher", page_icon="🤖")

st.title("🤖 AI-Powered Resume & Job Matching System")
st.write("Upload your resume(s) and a job description.")

col1,col2=st.columns(2)

with col1:
    job_description=st.text_area("Job Description")

with col2:
    uploaded_files=st.file_uploader(
        "Upload Resume(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

resume_texts={}
if uploaded_files:
    with st.spinner("Processing resumes..."):
        resume_texts=process_and_store_resumes(uploaded_files)
    st.success(f"{len(uploaded_files)} resume(s) processed.")

c1,c2=st.columns(2)
eval_btn=c1.button("Evaluate Resume")
match_btn=c2.button("ATS Match")

if eval_btn:
    if not uploaded_files:
        st.warning("Upload a resume first.")
    else:
        for name,text in resume_texts.items():
            st.subheader(name)
            st.write(evaluate_resume(job_description,text))

if match_btn:
    if not uploaded_files:
        st.warning("Upload a resume first.")
    else:
        for name,text in resume_texts.items():
            st.subheader(name)
            st.write(ats_match(job_description,text))
