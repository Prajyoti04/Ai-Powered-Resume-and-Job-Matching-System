from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "resume-index"

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2"),
    )

index = pc.Index(INDEX_NAME)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def extract_resume_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def process_and_store_resumes(uploaded_files):
    resumes = {}
    for file in uploaded_files:
        text = extract_resume_text(file)
        vector = embedding_model.encode(text).tolist()
        index.upsert(vectors=[(file.name, vector, {"text": text})])
        resumes[file.name] = text
    return resumes

def evaluate_resume(job_description, resume_text):
    prompt=f"""You are an expert HR recruiter.
Review the resume against the job description.

Job Description:
{job_description}

Resume:
{resume_text}

Return:
1. Strengths
2. Weaknesses
3. Missing Skills
4. Suggestions
"""
    return gemini_model.generate_content(prompt).text

def ats_match(job_description, resume_text):
    prompt=f"""Compare the resume with the job description.

Job Description:
{job_description}

Resume:
{resume_text}

Return:
ATS Score: xx%
Matching Skills
Missing Keywords
Final Recommendation
"""
    return gemini_model.generate_content(prompt).text
