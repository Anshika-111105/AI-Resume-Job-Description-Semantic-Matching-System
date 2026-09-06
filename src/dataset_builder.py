import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DATA_DIR,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    LABEL2ID,
)
from src.data_loader import load_raw_dataset
from src.preprocessing import clean_text, format_resume_text
from src.utils import get_logger, set_seed, save_json

logger = get_logger("dataset_builder")

# Role Family Ontology for realistic pairing
ROLE_FAMILIES = {
    "Software Engineering": [
        "Software Engineer",
        "Backend Engineer",
        "Full Stack Engineer",
        "Technical Support Specialist",
    ],
    "Data & Analytics": [
        "Data Analyst",
        "BI Analyst",
        "Financial Analyst",
        "FP&A Analyst",
        "Business Analyst",
    ],
    "Marketing & Growth": [
        "Marketing Manager",
        "Content Marketer",
        "Performance Marketer",
    ],
    "Sales & Client Success": [
        "Sales Representative",
        "Account Executive",
        "Business Development Manager",
        "Customer Success Associate",
        "Customer Support Specialist",
    ],
    "Product Management": [
        "Product Manager",
        "Associate Product Manager",
        "Technical Product Manager",
    ],
    "Operations & Finance": [
        "Operations Manager",
        "Project Manager",
        "Program Coordinator",
        "Junior Accountant",
    ],
}

# Invert mapping: role -> family
ROLE_TO_FAMILY = {}
for fam, roles in ROLE_FAMILIES.items():
    for r in roles:
        ROLE_TO_FAMILY[r] = fam

# Curated Job Description Templates
JD_RESPONSIBILITIES = {
    "Software Engineer": [
        "Design, build, and maintain scalable software applications and modular architectures.",
        "Write clean, testable, and efficient code adhering to OOP best practices and unit testing.",
        "Collaborate with cross-functional teams to integrate REST APIs and database layers.",
        "Participate in code reviews, CI/CD pipeline automation, and containerized deployments.",
    ],
    "Backend Engineer": [
        "Architect and implement robust backend services, microservices, and database schemas.",
        "Optimize RESTful APIs and asynchronous data pipelines for high throughput and low latency.",
        "Ensure system security, database integrity, and automated unit/integration testing.",
        "Deploy and monitor containerized microservices using Docker and CI/CD tools.",
    ],
    "Full Stack Engineer": [
        "Develop end-to-end web applications across modern frontend and backend architectures.",
        "Integrate dynamic user interfaces with robust backend APIs and relational databases.",
        "Implement automated testing, version control workflows with Git, and continuous deployment.",
        "Collaborate with product managers and UX designers to deliver user-centric features.",
    ],
    "Data Analyst": [
        "Analyze complex datasets to identify trends, performance metrics, and business insights.",
        "Develop and maintain automated dashboards and reports using Power BI, Tableau, and SQL.",
        "Clean, transform, and model data using Python, Pandas, and ETL pipelines.",
        "Design and evaluate A/B testing experiments to support data-driven decision making.",
    ],
    "BI Analyst": [
        "Translate business requirements into interactive BI dashboards and reporting solutions.",
        "Query relational databases using SQL and perform exploratory analysis in Python/Excel.",
        "Track key business KPIs and deliver actionable forecasting models to executive teams.",
        "Partner with data engineers to optimize reporting data models and ETL workflows.",
    ],
    "Financial Analyst": [
        "Perform financial modeling, forecasting, variance analysis, and budget planning.",
        "Analyze financial statements, operational KPIs, and market trends to advise leadership.",
        "Prepare quarterly and annual management reporting packages in Excel and BI tools.",
        "Evaluate capital expenditure requests and investment opportunities.",
    ],
    "FP&A Analyst": [
        "Drive corporate financial planning, annual budgeting cycles, and monthly forecast updates.",
        "Build detailed financial models to assess strategic initiatives and business performance.",
        "Analyze revenue and expense drivers across business units to optimize financial margins.",
        "Prepare executive-level presentations on financial health and variance analysis.",
    ],
    "Business Analyst": [
        "Elicit, analyze, and document business requirements, PRDs, and process workflows.",
        "Bridge the gap between business stakeholders and technical development teams.",
        "Analyze operational data to identify bottlenecks and recommend workflow improvements.",
        "Support user acceptance testing (UAT) and measure post-implementation business impact.",
    ],
    "Marketing Manager": [
        "Develop and execute multi-channel marketing strategies to drive lead generation and brand growth.",
        "Oversee paid advertising campaigns across Google Ads, Meta Ads, and social platforms.",
        "Optimize landing page conversion rates and conduct systematic A/B testing.",
        "Analyze marketing analytics and attribution models to maximize ROI and customer acquisition.",
    ],
    "Content Marketer": [
        "Create compelling written content, blogs, whitepapers, and marketing copy for target audiences.",
        "Develop SEO-driven content strategies to increase organic search traffic and engagement.",
        "Collaborate with designers to produce landing pages and multi-format promotional campaigns.",
        "Track content performance metrics, engagement rates, and lead conversion rates.",
    ],
    "Performance Marketer": [
        "Manage, scale, and optimize performance marketing budgets across digital ad networks.",
        "Design landing page funnels, conversion optimization experiments, and creative tests.",
        "Monitor customer acquisition costs (CAC), return on ad spend (ROAS), and lifetime value.",
        "Utilize advanced analytics to uncover audience segments and attribution insights.",
    ],
    "Sales Representative": [
        "Generate qualified sales leads through outbound prospecting, cold outreach, and referrals.",
        "Conduct discovery calls, product demonstrations, and value-based sales presentations.",
        "Manage sales pipeline in CRM, nurture prospects, and achieve monthly quota targets.",
        "Collaborate with customer success teams to facilitate seamless onboarding.",
    ],
    "Account Executive": [
        "Lead the complete enterprise sales cycle from initial qualification to contract closing.",
        "Negotiate pricing terms, navigate complex stakeholder landscapes, and exceed revenue targets.",
        "Develop territory sales strategies and build long-term relationships with key decision-makers.",
        "Partner with sales engineers to deliver customized product demonstrations.",
    ],
    "Business Development Manager": [
        "Identify and establish strategic business partnerships, channel alliances, and growth vectors.",
        "Conduct market research to identify untapped industries, client segments, and revenue streams.",
        "Structure, negotiate, and close high-impact commercial agreements.",
        "Work with executive leadership to refine go-to-market strategies.",
    ],
    "Customer Success Associate": [
        "Manage client relationships post-sale to ensure high adoption, satisfaction, and retention.",
        "Conduct onboarding training, quarterly business reviews, and proactive health checks.",
        "Identify upsell and cross-sell opportunities to expand account revenue.",
        "Serve as the client advocate by relaying product feedback to engineering and product teams.",
    ],
    "Customer Support Specialist": [
        "Provide timely, empathetic technical support and issue resolution across chat and email.",
        "Troubleshoot user problems, log bug tickets, and escalate technical issues to engineering.",
        "Create and maintain user-facing help documentation and knowledge base articles.",
        "Maintain high customer satisfaction (CSAT) scores and resolution efficiency.",
    ],
    "Technical Support Specialist": [
        "Diagnose and resolve complex technical, software, and database issues for enterprise users.",
        "Analyze log files, query databases with SQL, and replicate software bugs for developers.",
        "Provide tier-2 and tier-3 support, incident escalation, and root cause analysis.",
        "Document troubleshooting procedures and collaborate with development teams.",
    ],
    "Product Manager": [
        "Define product vision, roadmap, and quarterly prioritization aligned with company goals.",
        "Write detailed Product Requirement Documents (PRDs) and user stories for engineering.",
        "Conduct user research, customer interviews, and usability testing to validate features.",
        "Analyze product analytics, conversion funnels, and KPIs to drive continuous iteration.",
    ],
    "Associate Product Manager": [
        "Assist in creating PRDs, defining user stories, and managing sprint backlogs.",
        "Perform market analysis, competitor benchmarking, and user feedback synthesis.",
        "Track product usage metrics, feature adoption, and user satisfaction scores.",
        "Coordinate with engineering and design teams throughout the development lifecycle.",
    ],
    "Technical Product Manager": [
        "Drive technical product strategy for core infrastructure, APIs, and data platforms.",
        "Collaborate closely with software architects and engineers on system scalability.",
        "Define technical requirements, API specifications, and developer documentation.",
        "Bridge technical constraints with business objectives and product milestones.",
    ],
    "Operations Manager": [
        "Oversee daily business operations, resource allocation, and workflow optimization.",
        "Establish operational KPIs, performance dashboards, and quality control standards.",
        "Lead cross-departmental efficiency initiatives and cost-reduction projects.",
        "Manage vendor contracts, operational budgets, and process automation efforts.",
    ],
    "Project Manager": [
        "Lead project planning, milestone tracking, risk management, and resource scheduling.",
        "Facilitate Agile ceremonies, sprint planning, daily standups, and retrospectives.",
        "Ensure projects are delivered on time, within budget, and meeting quality specifications.",
        "Communicate project progress, roadblocks, and KPI updates to key stakeholders.",
    ],
    "Program Coordinator": [
        "Coordinate multi-track organizational initiatives, cross-functional schedules, and logistics.",
        "Track program deliverables, maintain documentation, and organize stakeholder meetings.",
        "Monitor program budget expenditures and compile status reports for management.",
        "Support continuous operational improvement and team collaboration workflows.",
    ],
    "Junior Accountant": [
        "Assist with general ledger maintenance, journal entries, and account reconciliations.",
        "Process accounts payable and accounts receivable transactions accurately.",
        "Support month-end and year-end financial close processes and audit preparations.",
        "Maintain organized financial records and assist with compliance reporting.",
    ],
}


def generate_job_description(
    role: str,
    seniority: str,
    industry: str,
    required_skills: List[str],
    education_req: str = "BSc",
    min_years: int = 3,
) -> str:
    """
    Generate a realistic, comprehensive Job Description text.
    """
    resp_list = JD_RESPONSIBILITIES.get(
        role,
        [
            f"Deliver high-impact results in {role} functions.",
            "Collaborate effectively with cross-functional team members.",
            "Maintain operational excellence and data-driven reporting.",
        ],
    )
    responsibilities_text = "\n".join([f"- {r}" for r in resp_list])
    skills_text = ", ".join(required_skills)

    jd_text = (
        f"Job Title: {seniority} {role}\n"
        f"Industry: {industry}\n"
        f"Experience Level: {seniority} ({min_years}+ years of relevant experience)\n"
        f"Education Requirement: {education_req} or equivalent practical experience\n\n"
        f"About the Role:\n"
        f"We are seeking a talented and proactive {seniority} {role} to join our growing {industry} team. "
        f"In this role, you will apply your expertise to build high-quality solutions, drive innovation, and collaborate with cross-functional stakeholders.\n\n"
        f"Key Responsibilities:\n"
        f"{responsibilities_text}\n\n"
        f"Required Qualifications & Skills:\n"
        f"- Proficient in: {skills_text}\n"
        f"- Proven track record of delivering projects in fast-paced environments\n"
        f"- Strong communication, problem-solving, and analytical skills"
    )
    return clean_text(jd_text)


def build_candidate_job_pairs(
    candidates_df: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Generate balanced candidate-job pairs for a given split of candidate resumes:
    - 1/3 Good Fit (Label 2)
    - 1/3 Potential Fit (Label 1)
    - 1/3 No Fit (Label 0)

    Guarantees strict split isolation.
    """
    rng = random.Random(seed)
    pairs = []

    roles_list = list(JD_RESPONSIBILITIES.keys())

    # Pre-aggregate skills by role in this split
    role_typical_skills = {}
    for r in roles_list:
        sub = candidates_df[candidates_df["role"] == r]
        if len(sub) > 0:
            skills_pool = [s for sk_list in sub["skills"] for s in sk_list]
            from collections import Counter
            top_skills = [s for s, _ in Counter(skills_pool).most_common(8)]
            role_typical_skills[r] = top_skills
        else:
            role_typical_skills[r] = ["Problem Solving", "Communication", "Teamwork"]

    for idx, cand in candidates_df.iterrows():
        cand_role = cand["role"]
        cand_seniority = cand["seniority"]
        cand_industry = cand["industry"]
        cand_skills = cand["skills"] if isinstance(cand["skills"], list) else []
        cand_edu = cand["education"]
        cand_years = cand["years_experience"]
        cand_resume_text = format_resume_text(cand)
        cand_family = ROLE_TO_FAMILY.get(cand_role, "Other")

        # --- 1. Good Fit Pair (Label 2) ---
        # Same role, matching seniority, high skill overlap (>=70% skills match)
        good_skills = list(set(cand_skills[:4] + role_typical_skills.get(cand_role, [])[:3]))
        rng.shuffle(good_skills)
        good_jd = generate_job_description(
            role=cand_role,
            seniority=cand_seniority,
            industry=cand_industry,
            required_skills=good_skills,
            education_req=cand_edu,
            min_years=max(1, cand_years - 1),
        )
        pairs.append({
            "resume_id": cand["resume_id"],
            "candidate_role": cand_role,
            "candidate_seniority": cand_seniority,
            "candidate_skills": cand_skills,
            "resume_text": cand_resume_text,
            "job_role": cand_role,
            "job_seniority": cand_seniority,
            "job_description": good_jd,
            "label": LABEL2ID["Good Fit"],
            "label_name": "Good Fit",
        })

        # --- 2. Potential Fit Pair (Label 1) ---
        # Adjacent role in same family OR same role with seniority shift / 40% missing skills
        adjacent_roles = [r for r in ROLE_FAMILIES.get(cand_family, []) if r != cand_role]
        if adjacent_roles:
            pot_role = rng.choice(adjacent_roles)
        else:
            pot_role = cand_role

        # Partially shared skills (e.g. 2 candidate skills + 3 new skills)
        pot_required_skills = list(set(cand_skills[:2] + role_typical_skills.get(pot_role, [])[:4]))
        rng.shuffle(pot_required_skills)
        pot_seniority = "Senior" if cand_seniority == "Mid" else ("Mid" if cand_seniority == "Junior" else "Mid")
        pot_jd = generate_job_description(
            role=pot_role,
            seniority=pot_seniority,
            industry=cand_industry,
            required_skills=pot_required_skills,
            education_req=cand_edu,
            min_years=cand_years + 2,
        )
        pairs.append({
            "resume_id": cand["resume_id"],
            "candidate_role": cand_role,
            "candidate_seniority": cand_seniority,
            "candidate_skills": cand_skills,
            "resume_text": cand_resume_text,
            "job_role": pot_role,
            "job_seniority": pot_seniority,
            "job_description": pot_jd,
            "label": LABEL2ID["Potential Fit"],
            "label_name": "Potential Fit",
        })

        # --- 3. No Fit Pair (Label 0) ---
        # Pick a role from a completely different role family (<10% overlap)
        diff_families = [fam for fam in ROLE_FAMILIES.keys() if fam != cand_family]
        no_family = rng.choice(diff_families)
        no_role = rng.choice(ROLE_FAMILIES[no_family])
        no_skills = role_typical_skills.get(no_role, [])[:5]
        no_industry = "Healthcare" if cand_industry != "Healthcare" else "FinTech"

        no_jd = generate_job_description(
            role=no_role,
            seniority=cand_seniority,
            industry=no_industry,
            required_skills=no_skills,
            education_req="MSc",
            min_years=5,
        )
        pairs.append({
            "resume_id": cand["resume_id"],
            "candidate_role": cand_role,
            "candidate_seniority": cand_seniority,
            "candidate_skills": cand_skills,
            "resume_text": cand_resume_text,
            "job_role": no_role,
            "job_seniority": cand_seniority,
            "job_description": no_jd,
            "label": LABEL2ID["No Fit"],
            "label_name": "No Fit",
        })

    pairs_df = pd.DataFrame(pairs)
    # Shuffle pairs deterministically
    pairs_df = pairs_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return pairs_df


def build_and_save_dataset(
    sample_size_per_split: int = None,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete dataset pipeline:
    1. Load raw resumes.
    2. Split resumes into Train (70%), Val (15%), Test (15%) by resume_id with stratification.
    3. Generate 3-class candidate-job pairs independently per split.
    4. Save to data/processed/.

    Args:
        sample_size_per_split: Optional subset size for rapid local execution.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    set_seed(seed)
    raw_df = load_raw_dataset()

    logger.info(f"Loaded {len(raw_df)} candidate resumes.")

    # Stratified split by candidate role to guarantee balanced distribution across splits
    train_resumes, temp_resumes = train_test_split(
        raw_df,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=seed,
        stratify=raw_df["role"],
    )

    val_resumes, test_resumes = train_test_split(
        temp_resumes,
        test_size=0.5,
        random_state=seed,
        stratify=temp_resumes["role"],
    )

    logger.info(f"Resume splits: Train={len(train_resumes)}, Val={len(val_resumes)}, Test={len(test_resumes)}")

    # Generate pairs independently per split
    logger.info("Generating candidate-job pairs for Training split...")
    train_pairs = build_candidate_job_pairs(train_resumes, seed=seed)

    logger.info("Generating candidate-job pairs for Validation split...")
    val_pairs = build_candidate_job_pairs(val_resumes, seed=seed + 1)

    logger.info("Generating candidate-job pairs for Test split...")
    test_pairs = build_candidate_job_pairs(test_resumes, seed=seed + 2)

    # Save to data/processed/
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_pairs.to_csv(train_path, index=False)
    val_pairs.to_csv(val_path, index=False)
    test_pairs.to_csv(test_path, index=False)

    logger.info(f"Saved processed dataset splits to {PROCESSED_DATA_DIR}")

    summary = {
        "total_resumes": len(raw_df),
        "train_resumes": len(train_resumes),
        "val_resumes": len(val_resumes),
        "test_resumes": len(test_resumes),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "train_class_distribution": train_pairs["label_name"].value_counts().to_dict(),
        "val_class_distribution": val_pairs["label_name"].value_counts().to_dict(),
        "test_class_distribution": test_pairs["label_name"].value_counts().to_dict(),
        "seed": seed,
    }
    save_json(summary, PROCESSED_DATA_DIR / "dataset_summary.json")
    logger.info("Dataset summary saved successfully.")

    return train_pairs, val_pairs, test_pairs


if __name__ == "__main__":
    train_df, val_df, test_df = build_and_save_dataset()
    print("\nDataset Construction Completed Successfully!")
    print(f"Train Pairs: {len(train_df)}")
    print(f"Val Pairs:   {len(val_df)}")
    print(f"Test Pairs:  {len(test_df)}")
    print("\nTrain Label Distribution:\n", train_df["label_name"].value_counts())
