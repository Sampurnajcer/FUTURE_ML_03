
import streamlit as st
import pandas as pd
import re
import nltk
import spacy
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# DOWNLOAD NLTK (FIRST TIME ONLY)
# ==========================================
nltk.download('stopwords')

# ==========================================
# PAGE SETTINGS
# ==========================================
st.set_page_config(
    page_title="AI Resume Screening System",
    layout="wide"
)

st.title("📄 AI Resume Screening & Ranking System")
st.markdown("### Machine Learning Resume Screening using NLP")

# ==========================================
# LOAD SPACY
# ==========================================
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("spaCy model not installed.")
    st.code("python -m spacy download en_core_web_sm")
    st.stop()

# ==========================================
# STOPWORDS
# ==========================================
stop_words = set(stopwords.words('english'))

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():

    resumes = pd.read_csv("resumes.csv")
    jobs = pd.read_csv("jobs.csv")

    return resumes, jobs

try:
    resumes_df, jobs_df = load_data()
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()

# ==========================================
# SHOW COLUMNS
# ==========================================
st.sidebar.header("Dataset Configuration")

st.sidebar.write("Resume Dataset Columns")
st.sidebar.write(list(resumes_df.columns))

st.sidebar.write("Job Dataset Columns")
st.sidebar.write(list(jobs_df.columns))

# ==========================================
# COLUMN SELECTION
# ==========================================
resume_column = st.sidebar.selectbox(
    "Select Resume Text Column",
    resumes_df.columns
)

job_column = st.sidebar.selectbox(
    "Select Job Description Column",
    jobs_df.columns
)

# ==========================================
# TEXT CLEANING
# ==========================================
def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r'http\S+', ' ', text)

    text = re.sub(r'www\S+', ' ', text)

    text = re.sub(r'[^a-zA-Z#+ ]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    cleaned = " ".join(words)

    return cleaned

# ==========================================
# LARGE SKILLS DATABASE
# ==========================================
skills_list = [
    "python",
    "java",
    "c",
    "c++",
    "sql",
    "mysql",
    "mongodb",
    "machine learning",
    "deep learning",
    "nlp",
    "tensorflow",
    "keras",
    "pandas",
    "numpy",
    "matplotlib",
    "data analysis",
    "excel",
    "power bi",
    "tableau",
    "communication",
    "leadership",
    "teamwork",
    "html",
    "css",
    "javascript",
    "react",
    "nodejs",
    "flask",
    "django",
    "git",
    "github",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "linux",
    "data science",
    "artificial intelligence",
    "computer vision",
    "opencv",
    "statistics",
    "problem solving",
    "spring boot"
]

# ==========================================
# SKILL EXTRACTION
# ==========================================
def extract_skills(text):

    text = clean_text(text)

    found_skills = []

    for skill in skills_list:

        if skill.lower() in text:
            found_skills.append(skill)

    return list(set(found_skills))

# ==========================================
# JOB DESCRIPTION SECTION
# ==========================================
st.header("🧾 Job Description")

sample_job = jobs_df[job_column].dropna().astype(str).iloc[0]

job_description = st.text_area(
    "Paste Job Description",
    value=sample_job,
    height=250
)

# ==========================================
# START SCREENING
# ==========================================
if st.button("🔍 Screen Candidates"):

    if len(job_description.strip()) == 0:
        st.warning("Please enter a job description")
        st.stop()

    cleaned_job = clean_text(job_description)

    # ==========================================
    # EXTRACT JOB SKILLS
    # ==========================================
    job_skills = extract_skills(job_description)

    st.subheader("✅ Extracted Job Skills")

    if len(job_skills) > 0:
        st.success(", ".join(job_skills))
    else:
        st.warning("No skills extracted. Add more technical skills in job description.")

    # ==========================================
    # REMOVE EMPTY RESUMES
    # ==========================================
    resumes_df = resumes_df.dropna(subset=[resume_column])

    resumes_df["cleaned_resume"] = resumes_df[resume_column].astype(str).apply(clean_text)

    # ==========================================
    # CALCULATE SCORES
    # ==========================================
    results = []

    for index, row in resumes_df.iterrows():

        resume_text = row["cleaned_resume"]

        # Skip very small resumes
        if len(resume_text.strip()) < 20:
            continue

        try:
            vectorizer = TfidfVectorizer()

            vectors = vectorizer.fit_transform([
                cleaned_job,
                resume_text
            ])

            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

            score = round(similarity * 100, 2)

        except:
            score = 0

        # ==========================================
        # SKILLS
        # ==========================================
        resume_skills = extract_skills(resume_text)

        matched_skills = list(set(job_skills) & set(resume_skills))

        missing_skills = list(set(job_skills) - set(resume_skills))

        results.append({
            "Candidate ID": index + 1,
            "Match Score (%)": score,
            "Matched Skills": ", ".join(matched_skills),
            "Missing Skills": ", ".join(missing_skills),
            "Resume Skills": ", ".join(resume_skills)
        })

    # ==========================================
    # RESULTS DATAFRAME
    # ==========================================
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        st.error("No valid resumes found.")
        st.stop()

    results_df = results_df.sort_values(
        by="Match Score (%)",
        ascending=False
    )

    # ==========================================
    # DISPLAY RESULTS
    # ==========================================
    st.header("🏆 Candidate Rankings")

    st.dataframe(results_df, use_container_width=True)

    # ==========================================
    # TOP CANDIDATE
    # ==========================================
    top_candidate = results_df.iloc[0]

    st.subheader("🥇 Best Candidate")

    st.success(
        f"Candidate {top_candidate['Candidate ID']} scored {top_candidate['Match Score (%)']}%"
    )

    # ==========================================
    # VISUALIZATION
    # ==========================================
    st.subheader("📊 Candidate Score Visualization")

    fig, ax = plt.subplots(figsize=(8, 3))

    ax.bar(
        results_df["Candidate ID"].astype(str),
        results_df["Match Score (%)"]
    )

    ax.set_xlabel("Candidate ID")
    ax.set_ylabel("Match Score")

    st.pyplot(fig)

    # ==========================================
    # DOWNLOAD RESULTS
    # ==========================================
    csv = results_df.to_csv(index=False)

    st.download_button(
        label="⬇ Download Results CSV",
        data=csv,
        file_name="candidate_ranking_results.csv",
        mime="text/csv"
    )

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("### Built using Streamlit, NLP, TF-IDF & Cosine Similarity")
