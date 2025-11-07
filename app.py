from dotenv import load_dotenv
import mysql.connector  # Importing MySQL connector

# Reminder: Run this application using the command: streamlit run app.py
load_dotenv()
import base64
import streamlit as st
import os
import io
from PIL import Image 
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import numpy as np

import pdf2image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# MySQL connection setup
def create_db_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            username='newuser',
            password='newpassword',
            database='codewthme'
        )
        print("Connection successfully created!")
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Create a connection to MySQL
db_connection = create_db_connection()

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

# Streamlit App
st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")
st.markdown("<style>body { background-color: rgba(240, 242, 245, 0.8); backdrop-filter: blur(10px); border-radius: 10px; }</style>", unsafe_allow_html=True)

st.markdown("<style>body { font-family: 'Arial', sans-serif; font-weight: bold; }</style>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    input_text = st.text_area("Job Description: ", key="input")
with col2:
    uploaded_files = st.file_uploader("Upload your resumes (PDF)...", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) Uploaded Successfully", icon="✅")

    with st.spinner("Processing resumes..."):
        process_and_store_resumes(uploaded_files)
        st.balloons()  # Add balloons animation during processing

submit1 = st.button("Tell Me About the Resume", key="submit1", help="Click to get insights about the resume.", icon="📄")

submit3 = st.button("Percentage Match", key="submit3", help="Click to get the percentage match of the resume.", icon="📊")

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
    # Prepare data for visualization
    matches = []  # List to hold match percentages
    keywords_found = []  # List to hold keywords found
    keywords_missing = []  # List to hold keywords missing

    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_prompt1, pdf_content, input_text)            
            match_percentage = ...  # Calculate match percentage based on response
            keywords_found = ...  # Extract keywords found from response
            keywords_missing = ...  # Extract keywords missing from response

            st.subheader(f"Response for {uploaded_file.name}:")
            st.write(response)

        # Visualization of results using Plotly
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Match Percentage', 'Keywords Found', 'Keywords Missing'],
            y=[match_percentage, len(keywords_found), len(keywords_missing)],
            marker_color=['green', 'blue', 'red']
        ))
        fig.update_layout(
            title='Resume Evaluation Results',
            yaxis_title='Count'
        )
        st.plotly_chart(fig)


    else:
        st.warning("Please upload the resume")

elif submit3:
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_prompt3, pdf_content, input_text)
            st.subheader(f"Response for {uploaded_file.name}:")
            st.write(response)

    else:
        st.warning("Please upload the resume")
