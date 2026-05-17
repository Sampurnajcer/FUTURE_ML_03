# AI Resume Screening & Candidate Ranking System
An NLP-based Resume Screening System that automatically analyzes resumes, extracts skills, compares them with job descriptions, and ranks candidates based on relevance.

## Project Objective
This project simulates a real-world Applicant Tracking System (ATS) used by companies to automate resume screening and reduce recruiter workload.

## Features
- Resume text preprocessing using NLP
- Skill extraction from resumes and job descriptions
- TF-IDF vectorization and Cosine Similarity scoring
- Candidate ranking based on job fit
- Missing skill identification
- Interactive Streamlit web interface
- Candidate score visualization

## Technologies Used
- Python
- Streamlit
- Pandas
- NLTK
- spaCy
- Scikit-learn
- Matplotlib

## How It Works
1. Upload resume and job description datasets
2. Enter a job description
3. System extracts important skills
4. Resumes are compared with the job role
5. Candidates are ranked based on similarity score

## Output
- Match Score (%)
- Matched Skills
- Missing Skills
- Ranked Candidate List
- Graphical Visualization

## Run the Project
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the application:
```bash
py -m streamlit run app.py
```



