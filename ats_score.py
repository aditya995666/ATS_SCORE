from flask import Flask, request, jsonify
import os
import re
import pandas as pd
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util
import torch

# ✅ Avoid GPU delay if running on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Constants
UPLOAD_FOLDER = 'resume'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Load model once
print("⏳ Loading model...")
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded.")

# ✅ Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ✅ Extract and clean resume text
def extract_resume_text(pdf_path):
    try:
        text = extract_text(pdf_path)
        return clean_text(text)
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
        return ""

# ✅ Job descriptions (your existing dictionary)
from job_descriptions import job_descriptions  # make sure this file exists and is clean

# ✅ Semantic Match
def get_job_matches(resume_text, top_n=3):
    job_titles = list(job_descriptions.keys())
    job_texts = [clean_text(desc) for desc in job_descriptions.values()]

    try:
        job_embeddings = MODEL.encode(job_texts, convert_to_tensor=True)
        resume_embedding = MODEL.encode(resume_text, convert_to_tensor=True)
        similarity_scores = util.cos_sim(resume_embedding, job_embeddings)[0].cpu().tolist()
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return []

    top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:top_n]
    matches = []

    for rank, idx in enumerate(top_indices, 1):
        title = job_titles[idx]
        score = similarity_scores[idx] * 100
        remark = "✅ Excellent" if score >= 75 else "⚠️ Average" if score >= 50 else "❌ Low"
        matches.append({
            "rank": rank,
            "job_title": title,
            "score": f"{score:.2f}",
            "remark": remark
        })

    return matches

# ✅ API route
@app.route('/match_resume', methods=['POST'])
def match_resume():
    if 'resume' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        resume_text = extract_resume_text(file_path)
        if not resume_text.strip():
            return jsonify({"error": "Unable to extract or clean resume text"}), 400

        results = get_job_matches(resume_text)
        return jsonify({
            "resume": file.filename,
            "matches": results
        })

    return jsonify({"error": "Invalid file format. Only PDF allowed."}), 400

# ✅ Root route
@app.route('/')
def index():
    return "✅ Resume Matcher API is running!"

# ✅ Run app
if __name__ == '__main__':
    app.run(debug=True)
