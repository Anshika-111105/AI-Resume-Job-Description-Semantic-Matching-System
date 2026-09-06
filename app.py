import os
import json
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Ensure src modules can be imported
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DISTILBERT_MODEL_DIR,
    ID2LABEL,
    LABEL2ID,
    LABEL_COLORS,
    SAMPLE_DATA_DIR,
)
from src.inference import ResumeJobMatcher
from src.resume_parser import extract_resume_text
from src.skill_extractor import analyze_skill_gap

# Page configuration
st.set_page_config(
    page_title="AI Resume–Job Description Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for polished, modern UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 1.75rem 2rem;
        border-radius: 12px;
        color: #F8FAFC;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 0.35rem;
        font-weight: 400;
    }
    
    .metric-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
    }
    
    .badge-good {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    .badge-potential {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid #F59E0B;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    .badge-nofit {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    .skill-chip {
        display: inline-block;
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 0.3rem 0.65rem;
        margin: 0.25rem;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .skill-match {
        border-left: 3px solid #10B981;
        color: #E2E8F0;
    }
    
    .skill-miss {
        border-left: 3px solid #EF4444;
        color: #CBD5E1;
    }
    
    .rec-card {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        color: #E2E8F0;
        font-size: 0.95rem;
    }
    
    .disclaimer-box {
        background-color: #1E293B;
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 0.85rem;
        font-size: 0.8rem;
        color: #94A3B8;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_matcher():
    """Cache the model inference engine across Streamlit re-runs."""
    return ResumeJobMatcher(model_dir=DISTILBERT_MODEL_DIR)


# Load Sample Data
@st.cache_data
def get_sample_data():
    sample_jds = {}
    sample_resumes = {}

    jds_file = SAMPLE_DATA_DIR / "sample_job_descriptions.json"
    if jds_file.exists():
        with open(jds_file, "r", encoding="utf-8") as f:
            sample_jds = json.load(f)

    ds_txt = SAMPLE_DATA_DIR / "resume_data_scientist.txt"
    if ds_txt.exists():
        sample_resumes["Data Scientist (TXT)"] = ds_txt.read_text(encoding="utf-8")

    docx_file = SAMPLE_DATA_DIR / "resume_software_engineer.docx"
    if docx_file.exists():
        try:
            sample_resumes["Software Engineer (DOCX)"] = extract_resume_text(docx_file)
        except Exception:
            pass

    pdf_file = SAMPLE_DATA_DIR / "resume_marketing_manager.pdf"
    if pdf_file.exists():
        try:
            sample_resumes["Marketing Manager (PDF)"] = extract_resume_text(pdf_file)
        except Exception:
            pass

    return sample_jds, sample_resumes


sample_jds, sample_resumes = get_sample_data()

# Header
st.markdown(
    """
    <div class="main-header">
        <h1 class="main-title">AI Resume–Job Description Matcher</h1>
        <div class="main-subtitle">Fine-Tuned DistilBERT Semantic Compatibility & Skill Gap Engine</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.subheader("System Overview")
    st.markdown(
        """
        This application uses a fine-tuned **DistilBERT** Transformer to compute deep semantic relevance 
        between candidate resumes and job descriptions, surpassing keyword-only ATS matchers.
        """
    )
    
    st.divider()
    st.subheader("Preset Samples")
    selected_sample = st.selectbox(
        "Load Sample Pair:",
        ["None (Custom)", "Data Scientist vs ML Engineer", "Software Engineer vs Backend Engineer", "Marketing Manager vs Marketing Role"],
    )

    st.divider()
    st.subheader("Model Status")
    matcher = load_matcher()
    st.success(f"Active Engine: `{matcher.model_type.upper()}`")
    st.info(f"Compute Device: `{matcher.device}`")

    st.divider()
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Ethical AI Notice:</strong><br>
            This system provides AI-generated semantic compatibility scores for guidance, research, and candidate self-assessment. 
            It is not intended for automated hiring or employment decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Input Section (2 Columns)
col_resume, col_jd = st.columns(2)

# Determine default texts from sample selector
default_resume_text = ""
default_jd_text = ""

if selected_sample == "Data Scientist vs ML Engineer":
    default_resume_text = sample_resumes.get("Data Scientist (TXT)", "")
    default_jd_text = sample_jds.get("Senior Machine Learning Engineer", "")
elif selected_sample == "Software Engineer vs Backend Engineer":
    default_resume_text = sample_resumes.get("Software Engineer (DOCX)", "")
    default_jd_text = sample_jds.get("Senior Backend Engineer", "")
elif selected_sample == "Marketing Manager vs Marketing Role":
    default_resume_text = sample_resumes.get("Marketing Manager (PDF)", "")
    default_jd_text = sample_jds.get("Digital Marketing Manager", "")

with col_resume:
    st.subheader("1. Candidate Resume")
    resume_input_mode = st.radio("Resume Input Method:", ["Upload File (PDF / DOCX / TXT)", "Paste Resume Text"], horizontal=True)

    extracted_resume_content = ""
    if resume_input_mode == "Upload File (PDF / DOCX / TXT)":
        uploaded_file = st.file_uploader("Upload resume file", type=["pdf", "docx", "txt"], help="Supported: PDF, DOCX, TXT")
        if uploaded_file is not None:
            try:
                extracted_resume_content = extract_resume_text(uploaded_file, uploaded_file.name)
                st.success(f"Successfully extracted `{uploaded_file.name}` ({len(extracted_resume_content.split())} words)")
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")
        elif default_resume_text:
            extracted_resume_content = default_resume_text
            st.info("Using pre-loaded sample resume.")
    else:
        extracted_resume_content = st.text_area(
            "Paste full resume content:",
            value=default_resume_text,
            height=260,
            placeholder="Paste candidate resume text here...",
        )

    if extracted_resume_content:
        with st.expander("Preview Parsed Resume Content"):
            st.text(extracted_resume_content[:1000] + ("..." if len(extracted_resume_content) > 1000 else ""))

with col_jd:
    st.subheader("2. Job Description")
    jd_input_content = st.text_area(
        "Paste Job Description text:",
        value=default_jd_text,
        height=310,
        placeholder="Paste target job description and requirements here...",
    )

st.write("")
analyze_col, _ = st.columns([1, 3])
with analyze_col:
    analyze_btn = st.button("🚀 Analyze Compatibility", type="primary", use_container_width=True)

# Analysis execution and results rendering
if analyze_btn or (selected_sample != "None (Custom)" and extracted_resume_content and jd_input_content):
    if not extracted_resume_content.strip():
        st.warning("Please provide or upload a candidate resume before analyzing.")
    elif not jd_input_content.strip():
        st.warning("Please provide a job description before analyzing.")
    else:
        with st.spinner("Analyzing semantic compatibility with DistilBERT..."):
            try:
                res = matcher.predict(extracted_resume_content, jd_input_content)
            except Exception as e:
                st.error(f"Inference error: {e}")
                st.stop()

        st.divider()
        st.subheader("Match Analysis Results")

        # Top Metric Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            score_color = "#10B981" if res["label"] == "Good Fit" else ("#F59E0B" if res["label"] == "Potential Fit" else "#EF4444")
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">Model Match Score</div>
                    <div class="metric-value" style="color: {score_color};">{res["score_percentage"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m_col2:
            badge_class = "badge-good" if res["label"] == "Good Fit" else ("badge-potential" if res["label"] == "Potential Fit" else "badge-nofit")
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">Category</div>
                    <div style="margin-top: 0.6rem;">
                        <span class="{badge_class}">{res["label"].upper()}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m_col3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">Model Confidence</div>
                    <div class="metric-value" style="color: #60A5FA;">{int(res["confidence"] * 100)}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m_col4:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">Inference Latency</div>
                    <div class="metric-value" style="color: #A78BFA;">{res["latency_ms"]} <span style="font-size:1rem;color:#94A3B8;">ms</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # Visual Probability Breakdown & Skill Stats
        chart_col, stat_col = st.columns([3, 2])

        with chart_col:
            st.markdown("#### Probability Distribution")
            probs = res["probabilities"]
            labels = list(probs.keys())
            values = [v * 100 for v in probs.values()]
            bar_colors = [LABEL_COLORS[LABEL2ID[k]] for k in labels]

            fig = go.Figure(
                go.Bar(
                    x=values,
                    y=labels,
                    orientation="h",
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    text=[f"{v:.1f}%" for v in values],
                    textposition="auto",
                    textfont=dict(color="white", size=13, family="Inter"),
                )
            )
            fig.update_layout(
                xaxis=dict(title="Probability (%)", range=[0, 100], showgrid=True, gridcolor="#334155"),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=10, b=20),
                height=220,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with stat_col:
            st.markdown("#### Skill Overlap Metric")
            skill_gap = res["skill_analysis"]
            st.markdown(
                f"""
                <div class="metric-box" style="text-align: left; padding: 1.1rem;">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #E2E8F0; margin-bottom: 0.5rem;">
                        Core Skill Coverage: <span style="color:#60A5FA;">{skill_gap["match_percentage"]}%</span>
                    </div>
                    <div style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
                        • <strong>Matching Skills:</strong> {skill_gap["matching_count"]} identified<br>
                        • <strong>Missing Skills:</strong> {skill_gap["missing_count"]} identified<br>
                        • <strong>Total Required:</strong> {skill_gap["total_required_count"]} detected in JD
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # Skill Breakdown Section
        s_col1, s_col2 = st.columns(2)

        with s_col1:
            st.markdown(f"#### ✅ Matching Skills ({skill_gap['matching_count']})")
            if skill_gap["matching_skills"]:
                chips_html = "".join([f'<span class="skill-chip skill-match">✓ {s}</span>' for s in skill_gap["matching_skills"]])
                st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)
            else:
                st.info("No direct skill overlaps detected in dictionary taxonomy.")

        with s_col2:
            st.markdown(f"#### ⚠️ Missing / Desired Skills ({skill_gap['missing_count']})")
            if skill_gap["missing_skills"]:
                chips_html = "".join([f'<span class="skill-chip skill-miss">• {s}</span>' for s in skill_gap["missing_skills"]])
                st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)
            else:
                st.success("Candidate matches all extracted technical skill requirements!")

        st.divider()

        # Actionable Recommendations
        st.markdown("#### 💡 Targeted Recommendations")
        for rec in skill_gap["recommendations"]:
            st.markdown(
                f"""
                <div class="rec-card">
                    📌 {rec}
                </div>
                """,
                unsafe_allow_html=True,
            )
