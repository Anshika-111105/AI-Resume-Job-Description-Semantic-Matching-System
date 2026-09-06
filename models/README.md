---
language:
- en
license: mit
library_name: transformers
tags:
- text-classification
- sequence-classification
- distilbert
- resume-matching
- job-description-matching
- semantic-search
- nlp
pipeline_tag: text-classification
metrics:
- accuracy
- f1
- precision
- recall
- roc_auc
widget:
- text: "Title: Senior Machine Learning Engineer\nSummary: 6 years experience in Python, PyTorch, Scikit-learn, and AWS."
  text_pair: "Job Title: Senior ML Engineer\nRequired Qualifications: Python, PyTorch, Docker, AWS cloud deployments."
  example_title: "Senior ML Engineer Match"
- text: "Title: Marketing Manager\nSummary: Experienced in Google Ads, SEO, and social media campaigns."
  text_pair: "Job Title: Backend Engineer\nRequired Qualifications: Python, PostgreSQL, Microservices, Kubernetes."
  example_title: "Cross-Domain No Fit"
---

# AI Resume–Job Description Semantic Matching System

## Model Description
This repository contains a **DistilBERT** (`distilbert-base-uncased`) transformer model fine-tuned for sequence pair classification on candidate resumes and job descriptions. 

Unlike traditional keyword-based Applicant Tracking Systems (ATS) that rely strictly on lexical overlap (BM25 / TF-IDF), this model captures deep semantic relationships, synonymy, domain equivalencies, and cross-functional skill alignments.

## Intended Use
- **Candidate Self-Assessment**: Analyze compatibility with target job descriptions and identify missing skills.
- **Recruiter Decision-Support**: Assist human recruiters in prioritizing relevant candidate profiles.
- **NLP Research & Education**: Demonstration of fine-tuning Transformers on structured-unstructured text pairs.

## Model Architecture
```text
Resume [SEP] Job Description
         ↓
  DistilBERT Encoder (6 Layers, 768 Hidden Dim, 12 Heads)
         ↓
  [CLS] Representation
         ↓
  Pre-Classifier Dense Layer (ReLU + Dropout 0.2)
         ↓
  Classification Head (3 Output Logits)
         ↓
  Softmax Probabilities: [No Fit, Potential Fit, Good Fit]
```

## Classes & Label Definitions
- **Class 0 (`No Fit`)**: Cross-domain mismatch (e.g. Sales Representative vs Software Engineer) with negligible skill or domain relevance.
- **Class 1 (`Potential Fit`)**: Adjacent role family or partial skill overlap (e.g. Data Analyst vs BI Analyst, or same role family with a seniority shift).
- **Class 2 (`Good Fit`)**: Target role family, high core skill coverage (≥70%), and aligned experience/seniority level.

## Training Details
- **Base Architecture**: `distilbert-base-uncased`
- **Source Dataset**: [`michaelozon/candidate-matching-synthetic`](https://huggingface.co/datasets/michaelozon/candidate-matching-synthetic) (10,000 candidate profiles, 24 roles, 73 skills)
- **Zero-Leakage Splitting**: Resumes partitioned by candidate ID *before* pair generation (70% Train, 15% Val, 15% Test)
- **Optimizer**: AdamW (Learning Rate: `2e-5`, Weight Decay: `0.01`)
- **Batch Size**: 8–16 with Warmup
- **Sequence Length**: 512 (Max Truncation)
- **Epochs**: 2–3

## Evaluation Results

| Model | Accuracy | Precision | Recall | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Cosine Similarity Baseline** | 0.9647 | 0.9647 | 0.9647 | 0.9647 | 0.9647 |
| **TF-IDF + Logistic Regression Baseline** | 0.9949 | 0.9949 | 0.9949 | 0.9949 | 0.9949 |
| **DistilBERT (Fine-Tuned Transformer)** | **0.99+** | **0.99+** | **0.99+** | **0.99+** | **0.99+** |

## How to Use with Transformers

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "anshika-saklani/resume-job-distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

resume = "Senior Python Engineer with 6 years experience in Django, REST APIs, Docker, and PostgreSQL."
job_desc = "Seeking Senior Backend Engineer with Python, Docker, Kubernetes, and REST API architecture experience."

inputs = tokenizer(
    resume,
    job_desc,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()

id2label = {0: "No Fit", 1: "Potential Fit", 2: "Good Fit"}
for idx, prob in enumerate(probs):
    print(f"{id2label[idx]}: {prob * 100:.2f}%")
```

## Ethical Considerations & Limitations
- **Synthetic Data**: The training corpus is derived from synthetic candidate profiles. Real-world resumes contain unstructured formatting, idiosyncrasies, and regional terminology differences.
- **Fairness & Non-Discrimination**: This tool evaluates lexical and semantic relevance only. It MUST NOT be used as an automated hiring decision-maker.
- **Truncation**: Inputs exceeding 512 subword tokens are truncated, which may omit details in multi-page resumes.
