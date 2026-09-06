import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd


def clean_text(text: Optional[str]) -> str:
    """
    Clean and normalize text for NLP processing and Transformer tokenization.

    Requirements:
    - Handle null/None/empty values gracefully.
    - Normalize unicode characters and remove non-printable control characters.
    - Normalize whitespace while preserving semantic sentence boundaries.
    - Preserve meaningful punctuation (e.g., C++, .NET, CI/CD, A/B Testing).
    - Do NOT stem or remove stopwords to retain contextual Transformer embeddings.

    Args:
        text (Optional[str]): Raw input text.

    Returns:
        str: Normalized, clean text string.
    """
    if text is None or not isinstance(text, str):
        return ""

    # Normalize unicode (NFKC canonical decomposition & composition)
    text = unicodedata.normalize("NFKC", text)

    # Remove non-printable control characters (except common whitespace \n, \t, \r)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch == "\r" or unicodedata.category(ch)[0] != "C")

    # Replace carriage returns and tabs with spaces
    text = text.replace("\r", " ").replace("\t", " ")

    # Collapse horizontal spaces
    text = re.sub(r"[^\S\n]+", " ", text)

    # Normalize spaces around newlines and collapse multiple newlines
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n+", "\n", text)

    # Strip leading and trailing whitespace
    return text.strip()


def prepare_text_pair(
    resume_text: Optional[str],
    job_description: Optional[str],
) -> Tuple[str, str]:
    """
    Clean and prepare a resume and job description pair for model input.

    Args:
        resume_text (Optional[str]): Candidate resume text.
        job_description (Optional[str]): Job description text.

    Returns:
        Tuple[str, str]: (cleaned_resume_text, cleaned_job_description)
    """
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(job_description)
    return clean_resume, clean_jd


def format_resume_text(row: Union[pd.Series, dict]) -> str:
    """
    Construct a complete, natural resume narrative from structured metadata fields.

    Args:
        row: Series or dict with keys 'role', 'seniority', 'years_experience',
             'industry', 'education', 'skills', 'summary', 'experience_bullets'.

    Returns:
        str: Cohesive, structured resume document text.
    """
    role = row.get("role", "")
    seniority = row.get("seniority", "")
    years = row.get("years_experience", 0)
    industry = row.get("industry", "")
    education = row.get("education", "")
    summary = row.get("summary", "")

    skills = row.get("skills", [])
    if isinstance(skills, str):
        import ast
        try:
            skills = ast.literal_eval(skills)
        except Exception:
            skills = [s.strip() for s in skills.split(",") if s.strip()]
    skills_str = ", ".join(skills) if isinstance(skills, list) else str(skills)

    bullets = row.get("experience_bullets", [])
    if isinstance(bullets, str):
        import ast
        try:
            bullets = ast.literal_eval(bullets)
        except Exception:
            bullets = [bullets]
    bullets_str = "\n".join([f"- {b}" for b in bullets]) if isinstance(bullets, list) else str(bullets)

    resume_doc = (
        f"Title: {seniority} {role}\n"
        f"Professional Summary: {summary}\n"
        f"Industry Experience: {industry} ({years} years)\n"
        f"Education: {education}\n"
        f"Core Competencies: {skills_str}\n"
        f"Professional Experience:\n{bullets_str}"
    )
    return clean_text(resume_doc)


def calculate_text_statistics(text: str) -> Dict[str, int]:
    """
    Calculate word count, character count, and line count for text analysis.
    """
    cleaned = clean_text(text)
    words = cleaned.split()
    return {
        "char_count": len(cleaned),
        "word_count": len(words),
        "line_count": len(cleaned.splitlines()) if cleaned else 0,
    }
