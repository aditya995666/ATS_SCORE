# 🔎 Project Explanation (Resume Matcher API)

This project is a Flask-based Resume Matcher. Its goal is to compare a candidate’s resume with predefined job descriptions and rank which jobs are the best fit.

## 🔧 Main Components:

# Flask Web API

Provides an endpoint /match_resume where a user can upload a resume in PDF format.

Returns a JSON response with the top matching jobs and their similarity scores.

Resume Text Extraction & Cleaning

Uses pdfminer to extract raw text from PDF resumes.

Cleans text using regex:

Removes emails, numbers, links, special characters, and extra spaces.

Converts everything to lowercase for consistency.

Semantic Matching (AI / NLP Part)

Uses Sentence Transformers (all-MiniLM-L6-v2) to convert both resumes and job descriptions into vector embeddings.

Measures cosine similarity between resume embedding and job description embeddings.

Ranks jobs by similarity score.

Job Description Dataset

A dictionary of job titles and their descriptions is imported from job_descriptions.py.

This acts as the "knowledge base" for comparison.

Scoring & Remarks

Scores are converted into percentages.

# Remark system:

✅ Excellent (≥ 75%)

⚠️ Average (50–75%)

❌ Low (< 50%)

# 🎯 Why is this useful?

For Recruiters → Quickly screen resumes against multiple job roles.

For Candidates → Check how well their resume matches different jobs.

For ATS Systems → Acts as a semantic layer on top of keyword-based systems.
