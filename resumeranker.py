import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

load_dotenv()

# Initialize AzureChatOpenAI
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.3,
    max_tokens=500,
)

st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title(" Resume Matcher: Rank Resumes Against Job Description")
st.write(
    "This app uses an AI model to rank resumes based on their match with a given job description (JD). "
    "Upload a JD and multiple resumes in PDF format to get started."
)


# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "".join(page.extract_text() or "" for page in reader.pages)

# Upload JD
st.subheader(" Upload Job Description (JD)")
jd_file = st.file_uploader("Upload Job Description PDF", type="pdf", key="jd")

# Upload multiple resumes
st.subheader(" Upload Resumes (Max: 10 PDFs)")
resume_files = st.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True, key="resumes")

if jd_file and resume_files:
    with st.spinner("Extracting and processing..."):
        jd_text = extract_text_from_pdf(jd_file)

        results = []

        for resume_file in resume_files:
            resume_text = extract_text_from_pdf(resume_file)

            prompt = ChatPromptTemplate.from_template(
                """
                You are a recruiter. Based on the following job description (JD) and candidate resume,
                rate the resume's match with the JD on a scale from 0 to 100, and justify the score.

                JD:
                {jd}

                Resume:
                {resume}

                Return output in the format:
                Score: <score>/100
                Justification: <one short paragraph>
                """
            )

            formatted_prompt = prompt.format_messages(jd=jd_text, resume=resume_text)
            response = llm(formatted_prompt)

            score_line = response.content.split("\n")[0]
            try:
                score = int(score_line.split(":")[1].strip().replace("/100", ""))
            except:
                score = 0

            results.append({
                "name": resume_file.name,
                "score": score,
                "response": response.content
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        st.success(" Resumes processed and ranked!")

        # Show results
        st.subheader(" Ranked Resumes")
        for idx, res in enumerate(results, 1):
            st.markdown(f"### {idx}. {res['name']} —  Score: {res['score']}/100")
            with st.expander("View Justification"):
                st.markdown(res["response"])
