from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SEED,
    ID2LABEL,
)
from src.preprocessing import clean_text
from src.utils import get_logger, set_seed

logger = get_logger("baseline_models")


def compute_cosine_similarity_matrix(v1, v2) -> np.ndarray:
    """Compute pairwise row cosine similarities between two sparse matrices."""
    from sklearn.metrics.pairwise import paired_cosine_distances
    dist = paired_cosine_distances(v1, v2)
    sim = 1.0 - dist
    return np.clip(sim, 0.0, 1.0)


def evaluate_tfidf_cosine_similarity(train_df: pd.DataFrame,test_df: pd.DataFrame,) -> Dict[str, any]:
    """
    Baseline 1: Pure TF-IDF Cosine Similarity.
    Builds TF-IDF representations for resumes and job descriptions, computes cosine similarity,
    calibrates thresholds on training data, and evaluates classification on test data.
    """
    logger.info("Evaluating Baseline 1: TF-IDF Cosine Similarity...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
    )

    # Fit vectorizer on all text in training split
    all_train_text = train_df["resume_text"].tolist() + train_df["job_description"].tolist()
    vectorizer.fit(all_train_text)

    # Transform test set
    test_resume_vecs = vectorizer.transform(test_df["resume_text"])
    test_jd_vecs = vectorizer.transform(test_df["job_description"])

    test_sims = compute_cosine_similarity_matrix(test_resume_vecs, test_jd_vecs)

    # Threshold calibration (using tertiles of train similarities)
    train_res_vecs = vectorizer.transform(train_df["resume_text"])
    train_jd_vecs = vectorizer.transform(train_df["job_description"])
    train_sims = compute_cosine_similarity_matrix(train_res_vecs, train_jd_vecs)

    t_low = float(np.percentile(train_sims, 33.33))
    t_high = float(np.percentile(train_sims, 66.66))
    logger.info(f"Calibrated TF-IDF similarity thresholds: Low={t_low:.4f}, High={t_high:.4f}")

    # Map similarities to 3 classes (0: No Fit, 1: Potential Fit, 2: Good Fit)
    test_preds = np.zeros(len(test_sims), dtype=int)
    test_preds[test_sims >= t_high] = 2
    test_preds[(test_sims >= t_low) & (test_sims < t_high)] = 1

    y_test = test_df["label"].values

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, test_preds, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, test_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, test_preds, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, test_preds)

    logger.info(f"TF-IDF Cosine Similarity -> Accuracy: {acc:.4f}, Macro F1: {f1_macro:.4f}, Weighted F1: {f1_weighted:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[ID2LABEL[i] for i in range(3)],
        yticklabels=[ID2LABEL[i] for i in range(3)],
    )
    plt.title("Confusion Matrix — TF-IDF Cosine Similarity Baseline", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_baseline_tfidf_cosine.png", dpi=300)
    plt.close()

    return {
        "model": "TF-IDF Cosine Similarity",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "threshold_low": t_low,
        "threshold_high": t_high,
    }


def train_and_evaluate_logistic_regression(train_df: pd.DataFrame,test_df: pd.DataFrame,) -> Tuple[Pipeline, Dict[str, any]]:
    """
    Baseline 2: TF-IDF + Logistic Regression.
    Concatenates resume and JD with separator, fits a TF-IDF + LogisticRegression pipeline,
    evaluates classification metrics, and saves the trained model artifact.
    """
    logger.info("Training Baseline 2: TF-IDF + Logistic Regression...")

    def create_pair_text(df: pd.DataFrame) -> List[str]:
        return [
            f"RESUME: {r} \n\n JOB_DESCRIPTION: {j}"
            for r, j in zip(df["resume_text"], df["job_description"])
        ]

    X_train = create_pair_text(train_df)
    y_train = train_df["label"].values

    X_test = create_pair_text(test_df)
    y_test = test_df["label"].values

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=15000,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
    ])

    pipeline.fit(X_train, y_train)

    test_preds = pipeline.predict(X_test)
    test_probs = pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, test_preds, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, test_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, test_preds, average="weighted", zero_division=0)

    report = classification_report(
        y_test,
        test_preds,
        target_names=[ID2LABEL[i] for i in range(3)],
        output_dict=True,
    )
    cm = confusion_matrix(y_test, test_preds)

    logger.info(f"TF-IDF + Logistic Regression -> Accuracy: {acc:.4f}, Macro F1: {f1_macro:.4f}, Weighted F1: {f1_weighted:.4f}")

    # Save trained model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / "logistic_regression_baseline.joblib"
    joblib.dump(pipeline, model_save_path)
    logger.info(f"Saved baseline logistic regression model to {model_save_path}")

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[ID2LABEL[i] for i in range(3)],
        yticklabels=[ID2LABEL[i] for i in range(3)],
    )
    plt.title("Confusion Matrix — TF-IDF + Logistic Regression", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_baseline_logistic.png", dpi=300)
    plt.close()

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    return pipeline, metrics


def run_baseline_benchmarks() -> pd.DataFrame:
    """Run all baseline benchmarks and save results to outputs/metrics/baseline_metrics.csv."""
    set_seed(SEED)

    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        from src.dataset_builder import build_and_save_dataset
        train_df, _, test_df = build_and_save_dataset()
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

    logger.info(f"Running baselines on Train ({len(train_df)} rows) and Test ({len(test_df)} rows)...")

    # Run Baseline 1
    res1 = evaluate_tfidf_cosine_similarity(train_df, test_df)

    # Run Baseline 2
    _, res2 = train_and_evaluate_logistic_regression(train_df, test_df)

    results = [
        {
            "Model": res1["model"],
            "Accuracy": round(res1["accuracy"], 4),
            "Precision": round(res1["precision"], 4),
            "Recall": round(res1["recall"], 4),
            "Macro F1": round(res1["f1_macro"], 4),
            "Weighted F1": round(res1["f1_weighted"], 4),
        },
        {
            "Model": res2["model"],
            "Accuracy": round(res2["accuracy"], 4),
            "Precision": round(res2["precision"], 4),
            "Recall": round(res2["recall"], 4),
            "Macro F1": round(res2["f1_macro"], 4),
            "Weighted F1": round(res2["f1_weighted"], 4),
        },
    ]

    metrics_df = pd.DataFrame(results)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "baseline_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved baseline metrics table to {metrics_path}")

    print("\n--- Baseline Models Performance Summary ---")
    print(metrics_df.to_string(index=False))

    return metrics_df


if __name__ == "__main__":
    run_baseline_benchmarks()
