import pytest
from src.skill_extractor import extract_skills, analyze_skill_gap, generate_skill_recommendations


def test_extract_skills_basic():
    text = "Candidate has hands-on experience in Python, SQL, Docker, and Git."
    skills = extract_skills(text)
    assert "Python" in skills
    assert "SQL" in skills
    assert "Docker" in skills
    assert "Git" in skills


def test_extract_skills_case_insensitivity():
    text = "python, PYTHON, PyThOn, sql, Sql, SQL, docker"
    skills = extract_skills(text)
    assert "Python" in skills
    assert "SQL" in skills
    assert "Docker" in skills
    # Check deduplication
    assert len([s for s in skills if s == "Python"]) == 1


def test_extract_skills_alias_normalization():
    text = "Experience with sklearn, postgres, and k8s."
    skills = extract_skills(text)
    assert "Scikit-learn" in skills
    assert "PostgreSQL" in skills
    assert "Kubernetes" in skills


def test_extract_skills_special_symbols():
    text = "Proficient in C++, C#, .NET, CI/CD, and A/B Testing."
    skills = extract_skills(text)
    assert "C++" in skills
    assert "C#" in skills
    assert ".NET" in skills
    assert "CI/CD" in skills
    assert "A/B Testing" in skills


def test_analyze_skill_gap():
    resume = "Data Analyst skilled in Python, SQL, Pandas, and Tableau."
    jd = "Seeking Data Analyst with Python, SQL, AWS, Docker, and Snowflake."

    gap = analyze_skill_gap(resume, jd)

    assert "Python" in gap["matching_skills"]
    assert "SQL" in gap["matching_skills"]
    assert "AWS" in gap["missing_skills"]
    assert "Docker" in gap["missing_skills"]
    assert gap["matching_count"] == 2
    assert gap["missing_count"] == 3
    assert len(gap["recommendations"]) > 0
