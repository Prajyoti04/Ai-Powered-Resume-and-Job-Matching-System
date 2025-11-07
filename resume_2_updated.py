from dotenv import load_dotenv

# Reminder: Run this application using the command: streamlit run app.py
load_dotenv()
import base64
import streamlit as st
import os
import io
from PIL import Image 
import pdf2image
from dotenv import load_dotenv
import google.generativeai as genai


# Reminder: Run this application using the command: streamlit run app.py
load_dotenv()
import base64
import streamlit as st
import os
import io
from PIL import Image 
import pdf2image
from dotenv import load_dotenv
import google.generativeai as genai
import pinecone  

import numpy as np
from sentence_transformers import SentenceTransformer

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index_name = "resume-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )
index = pc.Index(index_name)


def get_gemini_response(input, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')  
    response = model.generate_content([input, pdf_content[0], prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        # Convert the PDF to image
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]

        # Convert to bytes
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
        raise FileNotFoundError("No file uploaded or invalid PDF file.")

def process_and_store_resumes(uploaded_files):
    for uploaded_file in uploaded_files:
        # Read the content of the PDF file
        pdf_content = input_pdf_setup(uploaded_file)
        # Convert the content to a vector
        text = " ".join([part['data'] for part in pdf_content])  
        vector = model.encode(text)  
        index.upsert([(uploaded_file.name, vector.tolist())])  

# Streamlit App
st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")
st.markdown("<style>body { font-family: 'Arial', sans-serif; }</style>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    input_text = st.text_area("Job Description: ", key="input")
with col2:
    uploaded_files = st.file_uploader("Upload your resumes (PDF)...", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) Uploaded Successfully")
    with st.spinner("Processing resumes..."):
        process_and_store_resumes(uploaded_files)

submit1 = st.button("Tell Me About the Resume")
submit3 = st.button("Percentage match")

input_prompt1 = """
You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. Give me the percentage of match if the resume matches
the job description. First the output should come as percentage and then keywords missing and last final thoughts.
"""

if submit1:
    if uploaded_files:
        pdf_content = input_pdf_setup(uploaded_files[0])  
        response = get_gemini_response(input_text, pdf_content, input_prompt1)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.warning("Please upload the resume")

elif submit3:
    if uploaded_files:
        pdf_content = input_pdf_setup(uploaded_files[0])  
        response = get_gemini_response(input_text, pdf_content, input_prompt3)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.warning("Please upload the resume")
import pinecone  # Updated import for the new Pinecone package

import numpy as np
from sentence_transformers import SentenceTransformer

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")  # Replace with your environment
index_name = "resume-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)  # Dimension of the embeddings from the model
index = pinecone.Index(index_name)

def get_gemini_response(input, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model
    response = model.generate_content([input, pdf_content[0], prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        # Convert the PDF to image
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded or invalid PDF file.")

def process_and_store_resumes(uploaded_files):
    for uploaded_file in uploaded_files:
        # Read the content of the PDF file
        pdf_content = input_pdf_setup(uploaded_file)
        # Convert the content to a vector
        text = " ".join([part['data'] for part in pdf_content])  # Example of extracting text
        vector = model.encode(text)  # Generate vector embedding
        index.upsert([(uploaded_file.name, vector.tolist())])  # Store vector in Pinecone

# Streamlit App
st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")
st.markdown("<style>body { font-family: 'Arial', sans-serif; }</style>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    input_text = st.text_area("Job Description: ", key="input")
with col2:
    uploaded_files = st.file_uploader("Upload your resumes (PDF)...", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) Uploaded Successfully")
    with st.spinner("Processing resumes..."):
        process_and_store_resumes(uploaded_files)

submit1 = st.button("Tell Me About the Resume")
submit3 = st.button("Percentage match")

input_prompt1 = """
You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. Give me the percentage of match if the resume matches
the job description. First the output should come as percentage and then keywords missing and last final thoughts.
"""

if submit1:
    if uploaded_files:
        pdf_content = input_pdf_setup(uploaded_files[0])  # Example for the first uploaded file
        response = get_gemini_response(input_prompt1, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.warning("Please upload the resume")

elif submit3:
    if uploaded_files:
        pdf_content = input_pdf_setup(uploaded_files[0])  # Example for the first uploaded file
        response = get_gemini_response(input_prompt3, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.warning("Please upload the resume")
