"""
Skill Extractor & Skill Gap Analysis Module
AI Resume–Job Description Semantic Matching System

Features:
- Multi-category technical & domain skill taxonomy
- Robust regex matching with boundary protection for C++, .NET, CI/CD, etc.
- Alias canonicalization (e.g. sklearn -> scikit-learn, postgres -> postgresql)
- Skill gap analysis comparing candidate resumes against job descriptions
- Actionable rule-based candidate recommendations
"""

import re
from typing import Dict, List, Set, Tuple

# Comprehensive Categorized Skill Taxonomy
SKILL_TAXONOMY = {
    "Programming": [
        "Python",
        "Java",
        "C++",
        "C#",
        ".NET",
        "JavaScript",
        "TypeScript",
        "Go",
        "Rust",
        "Ruby",
        "PHP",
        "SQL",
        "HTML",
        "CSS",
        "Bash",
        "Shell",
        "R",
        "MATLAB",
        "Scala",
        "Kotlin",
        "Swift",
    ],
    "Machine Learning & AI": [
        "Machine Learning",
        "Deep Learning",
        "Scikit-learn",
        "PyTorch",
        "TensorFlow",
        "Keras",
        "XGBoost",
        "LightGBM",
        "Hugging Face",
        "Transformers",
        "NLP",
        "Natural Language Processing",
        "Computer Vision",
        "LLMs",
        "LangChain",
        "MLflow",
        "Neural Networks",
        "Reinforcement Learning",
        "Model Deployment",
    ],
    "Data & Analytics": [
        "Pandas",
        "NumPy",
        "SciPy",
        "Matplotlib",
        "Seaborn",
        "Power BI",
        "Tableau",
        "Excel",
        "Data Visualization",
        "ETL",
        "Data Pipelines",
        "Statistics",
        "A/B Testing",
        "Feature Engineering",
        "BigQuery",
        "Snowflake",
        "Spark",
        "Hadoop",
        "Looker",
        "Financial Modeling",
        "Forecasting",
        "Reporting",
        "KPIs",
    ],
    "Cloud & Infrastructure": [
        "AWS",
        "Azure",
        "GCP",
        "Google Cloud",
        "Cloud Computing",
        "EC2",
        "S3",
        "Lambda",
        "Serverless",
    ],
    "DevOps & Engineering Practices": [
        "Docker",
        "Kubernetes",
        "Git",
        "GitHub",
        "GitLab",
        "CI/CD",
        "Linux",
        "Terraform",
        "Jenkins",
        "Ansible",
        "REST APIs",
        "Microservices",
        "Unit Testing",
        "OOP",
        "Object Oriented Programming",
        "Agile",
        "Scrum",
        "JIRA",
    ],
    "Databases": [
        "MySQL",
        "PostgreSQL",
        "MongoDB",
        "Redis",
        "SQLite",
        "Cassandra",
        "Elasticsearch",
        "DynamoDB",
        "Oracle",
        "SQL Server",
        "Databases",
    ],
    "Business & Marketing": [
        "Product Management",
        "Product Strategy",
        "PRD",
        "Roadmap",
        "User Research",
        "Prioritization",
        "Growth Marketing",
        "Google Ads",
        "Meta Ads",
        "Copywriting",
        "Content Marketing",
        "Landing Pages",
        "Conversion Optimization",
        "Email Marketing",
        "Salesforce",
        "CRM",
        "Prospecting",
        "Lead Generation",
        "Client Retention",
    ],
}

# Alias mapping: non-canonical alias -> canonical skill name
SKILL_ALIASES = {
    "sklearn": "Scikit-learn",
    "scikit learn": "Scikit-learn",
    "scikitlearn": "Scikit-learn",
    "tf": "TensorFlow",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "torch": "PyTorch",
    "postgres": "PostgreSQL",
    "pgsql": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "k8s": "Kubernetes",
    "ms excel": "Excel",
    "microsoft excel": "Excel",
    "powerbi": "Power BI",
    "power-bi": "Power BI",
    "gcp": "GCP",
    "google cloud platform": "GCP",
    "aws": "AWS",
    "amazon web services": "AWS",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "continuous integration": "CI/CD",
    "rest api": "REST APIs",
    "rest apis": "REST APIs",
    "restful api": "REST APIs",
    "restful apis": "REST APIs",
    "oop": "OOP",
    "object oriented programming": "OOP",
    "unit test": "Unit Testing",
    "unit tests": "Unit Testing",
    "unit testing": "Unit Testing",
    "a/b test": "A/B Testing",
    "a/b testing": "A/B Testing",
    "ab testing": "A/B Testing",
    "llm": "LLMs",
    "llms": "LLMs",
    "large language models": "LLMs",
    "nlp": "NLP",
    "natural language processing": "NLP",
    "git": "Git",
    "github": "GitHub",
    "docker": "Docker",
    "sql": "SQL",
    "python": "Python",
    "java": "Java",
    "c++": "C++",
    "cpp": "C++",
    "c#": "C#",
    "csharp": "C#",
    ".net": ".NET",
    "dotnet": ".NET",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "tableau": "Tableau",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "mysql": "MySQL",
    "jira": "JIRA",
}

# Compile custom patterns for regex matching
def _compile_skill_patterns():
    patterns = []
    # 1. From Taxonomy
    all_taxonomy_skills = set()
    for cat, skills in SKILL_TAXONOMY.items():
        all_taxonomy_skills.update(skills)

    for skill in all_taxonomy_skills:
        escaped = re.escape(skill)
        # Handle special cases with trailing + or # or leading .
        if skill in ["C++", "C#", ".NET", "CI/CD", "A/B Testing"]:
            pattern = re.compile(rf"(?:\b|(?<=\s)){escaped}(?:\b|(?=\s)|(?=[,.;:!?]))", re.IGNORECASE)
        else:
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        patterns.append((pattern, skill))

    # 2. From Aliases
    for alias, canonical in SKILL_ALIASES.items():
        escaped = re.escape(alias)
        if alias in ["c++", "c#", ".net", "ci/cd", "a/b test", "a/b testing", "ab testing"]:
            pattern = re.compile(rf"(?:\b|(?<=\s)){escaped}(?:\b|(?=\s)|(?=[,.;:!?]))", re.IGNORECASE)
        else:
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        patterns.append((pattern, canonical))

    return patterns


COMPILED_SKILL_PATTERNS = _compile_skill_patterns()


def extract_skills(text: str) -> List[str]:
    """
    Extract and normalize technical and domain skills from raw or cleaned text.

    Args:
        text (str): Input text (resume or job description).

    Returns:
        List[str]: Alphabetically sorted, deduplicated list of canonical skill names.
    """
    if not text or not isinstance(text, str):
        return []

    found_skills: Set[str] = set()

    for pattern, canonical_name in COMPILED_SKILL_PATTERNS:
        if pattern.search(text):
            found_skills.add(canonical_name)

    return sorted(list(found_skills))


def generate_skill_recommendations(missing_skills: List[str]) -> List[str]:
    """
    Generate actionable, constructive recommendations for missing skills without fabricating experience.
    """
    if not missing_skills:
        return [
            "Your resume strongly aligns with the core skill requirements for this position!",
            "Consider highlighting specific project achievements and quantitative metrics for your top skills.",
        ]

    recommendations = []
    category_suggestions = {
        "Cloud": ["AWS", "Azure", "GCP", "Cloud Computing", "EC2", "S3", "Lambda"],
        "Containerization & DevOps": ["Docker", "Kubernetes", "CI/CD", "Terraform", "Jenkins", "Ansible"],
        "Databases & Backend": ["PostgreSQL", "MySQL", "MongoDB", "Redis", "REST APIs", "Microservices"],
        "Machine Learning": ["PyTorch", "TensorFlow", "Scikit-learn", "NLP", "LLMs", "MLflow", "Computer Vision"],
        "Data & BI": ["Power BI", "Tableau", "Pandas", "ETL", "A/B Testing", "BigQuery", "Snowflake"],
    }

    # Group missing skills by theme
    matched_categories = set()
    for cat, cat_skills in category_suggestions.items():
        overlap = [s for s in missing_skills if s in cat_skills]
        if overlap and cat not in matched_categories:
            matched_categories.add(cat)
            skills_str = ", ".join(overlap[:3])
            recommendations.append(
                f"Consider highlighting experience with {skills_str} if applicable to your practical background."
            )

    # Specific missing skills suggestions
    for skill in missing_skills[:4]:
        if not any(skill in rec for rec in recommendations):
            recommendations.append(
                f"Mention any academic coursework, personal projects, or certifications involving {skill}."
            )

    # General advice
    recommendations.append(
        "Ensure your experience bullet points quantify the impact of your technical contributions."
    )

    return recommendations[:5]


def analyze_skill_gap(
    resume_text: str,
    job_description: str,
) -> Dict[str, any]:
    """
    Perform a complete skill gap analysis between candidate resume and job description.

    Args:
        resume_text (str): Candidate resume string.
        job_description (str): Job description string.

    Returns:
        Dict: Structure with matching_skills, missing_skills, resume_skills,
              required_skills, match_percentage, and recommendations.
    """
    resume_skills = extract_skills(resume_text)
    required_skills = extract_skills(job_description)

    resume_set = set(resume_skills)
    required_set = set(required_skills)

    matching_skills = sorted(list(resume_set.intersection(required_set)))
    missing_skills = sorted(list(required_set.difference(resume_set)))

    total_required = len(required_skills)
    match_percentage = round((len(matching_skills) / total_required * 100), 1) if total_required > 0 else 100.0

    recommendations = generate_skill_recommendations(missing_skills)

    return {
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "resume_skills": resume_skills,
        "required_skills": required_skills,
        "matching_count": len(matching_skills),
        "missing_count": len(missing_skills),
        "total_required_count": total_required,
        "match_percentage": match_percentage,
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    sample_resume = "Software Engineer experienced in Python, SQL, Docker, Scikit-learn, and Git. Built REST APIs."
    sample_jd = "Looking for a Senior Software Engineer with Python, SQL, AWS, Kubernetes, Docker, and CI/CD experience."

    gap = analyze_skill_gap(sample_resume, sample_jd)
    print("--- Skill Gap Analysis Demo ---")
    print(f"Matching Skills ({gap['matching_count']}):", gap["matching_skills"])
    print(f"Missing Skills ({gap['missing_count']}):", gap["missing_skills"])
    print(f"Skill Match Percentage: {gap['match_percentage']}%")
    print("\nRecommendations:")
    for r in gap["recommendations"]:
        print(f"  - {r}")
