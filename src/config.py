"""
Global Configuration & Constants
AI Resume–Job Description Semantic Matching System
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

MODELS_DIR = BASE_DIR / "models"
DISTILBERT_MODEL_DIR = MODELS_DIR / "distilbert_resume_matcher"

OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# Ensure essential directories exist
for p in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR,
    DISTILBERT_MODEL_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)

# Dataset configuration
HF_DATASET_NAME = "michaelozon/candidate-matching-synthetic"
HF_HUB_MODEL_ID = "anshika-saklani/resume-job-distilbert"

# Random Seed
SEED = 42

# Label Definitions
# 0 = No Fit (Unrelated domain/skills)
# 1 = Potential Fit (Adjacent domain / partial skill overlap)
# 2 = Good Fit (High skill overlap, role & seniority match)
LABEL2ID = {
    "No Fit": 0,
    "Potential Fit": 1,
    "Good Fit": 2,
}

ID2LABEL = {
    0: "No Fit",
    1: "Potential Fit",
    2: "Good Fit",
}

NUM_LABELS = 3

LABEL_COLORS = {
    0: "#EF4444",  # Red
    1: "#F59E0B",  # Amber/Yellow
    2: "#10B981",  # Green
}

# Splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Transformer Hyperparameters
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 1
