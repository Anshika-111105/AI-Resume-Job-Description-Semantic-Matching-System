"""
Evaluation & Error Analysis Module
AI Resume–Job Description Semantic Matching System
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    DISTILBERT_MODEL_DIR,
    FIGURES_DIR,
    ID2LABEL,
    LABEL2ID,
    MAX_LENGTH,
    METRICS_DIR,
    MODELS_DIR,
    NUM_LABELS,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    SEED,
)
from src.utils import get_device, get_logger, save_json, set_seed

logger = get_logger("evaluate")


def predict_batch_distilbert(
    test_df: pd.DataFrame,
    model_dir: Path = DISTILBERT_MODEL_DIR,
    batch_size: int = 32,
    max_length: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run batched inference with DistilBERT on test pairs.
    Returns predicted classes and probability distributions.
    """
    device = get_device()
    logger.info(f"Loading DistilBERT from {model_dir} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    all_preds = []
    all_probs = []

    resumes = test_df["resume_text"].tolist()
    jds = test_df["job_description"].tolist()
    total = len(test_df)

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_resumes = resumes[i : i + batch_size]
            batch_jds = jds[i : i + batch_size]

            encoded = tokenizer(
                text=batch_resumes,
                text_pair=batch_jds,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_probs.append(probs)
            all_preds.append(preds)

    return np.concatenate(all_preds, axis=0), np.concatenate(all_probs, axis=0)


def evaluate_transformer(
    test_df: pd.DataFrame,
    preds: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, any]:
    """
    Compute comprehensive metrics and generate ROC, PR, and Confusion Matrix plots for Transformer.
    """
    y_test = test_df["label"].values

    acc = accuracy_score(y_test, preds)
    prec_weighted = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, preds, average="weighted", zero_division=0)

    # One-vs-Rest ROC-AUC
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_auc_ovr = roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")

    cm = confusion_matrix(y_test, preds)
    report = classification_report(
        y_test,
        preds,
        target_names=[ID2LABEL[i] for i in range(3)],
        output_dict=True,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[ID2LABEL[i] for i in range(3)],
        yticklabels=[ID2LABEL[i] for i in range(3)],
        cbar=False,
    )
    plt.title("Confusion Matrix — Fine-Tuned DistilBERT", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_distilbert.png", dpi=300)
    plt.close()

    # 2. ROC Curves (One-vs-Rest)
    plt.figure(figsize=(8, 6))
    colors = ["#EF4444", "#F59E0B", "#10B981"]
    for i in range(NUM_LABELS):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        class_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=colors[i],
            lw=2,
            label=f"{ID2LABEL[i]} (AUC = {class_auc:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("One-vs-Rest ROC Curves — DistilBERT", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=300)
    plt.close()

    # 3. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for i in range(NUM_LABELS):
        precision_c, recall_c, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        pr_auc = auc(recall_c, precision_c)
        plt.plot(
            recall_c,
            precision_c,
            color=colors[i],
            lw=2,
            label=f"{ID2LABEL[i]} (PR-AUC = {pr_auc:.3f})",
        )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("Precision-Recall Curves — DistilBERT", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curve.png", dpi=300)
    plt.close()

    return {
        "model": "DistilBERT (Fine-Tuned)",
        "accuracy": acc,
        "precision": prec_weighted,
        "recall": rec_weighted,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "roc_auc_ovr": roc_auc_ovr,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def perform_qualitative_error_analysis(
    test_df: pd.DataFrame,
    preds: np.ndarray,
    probs: np.ndarray,
    num_examples: int = 12,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Perform deep qualitative error analysis on test predictions.
    Identifies misclassifications and categorizes root causes.
    """
    y_test = test_df["label"].values
    error_indices = np.where(preds != y_test)[0]
    logger.info(f"Total test errors: {len(error_indices)} / {len(test_df)} ({len(error_indices)/len(test_df)*100:.2f}%)")

    errors_list = []
    selected_indices = error_indices[: min(num_examples, len(error_indices))]

    # If fewer errors than requested, sample all available error indices
    for idx in selected_indices:
        row = test_df.iloc[idx]
        true_label = int(y_test[idx])
        pred_label = int(preds[idx])
        conf = float(probs[idx][pred_label])

        # Categorize root cause
        cand_role = row.get("candidate_role", "")
        job_role = row.get("job_role", "")
        cand_sen = row.get("candidate_seniority", "")
        job_sen = row.get("job_seniority", "")

        if cand_role == job_role and cand_sen != job_sen:
            category = "Seniority Mismatch / Ambiguity"
        elif cand_role != job_role and (true_label == 1 or pred_label == 1):
            category = "Adjacent Role Skill Overlap"
        elif len(row["resume_text"]) > 1200 or len(row["job_description"]) > 1000:
            category = "Document Length / Truncation Boundary"
        else:
            category = "Cross-Domain Semantic Nuance"

        errors_list.append({
            "index": int(idx),
            "candidate_role": cand_role,
            "job_role": job_role,
            "true_label": ID2LABEL[true_label],
            "predicted_label": ID2LABEL[pred_label],
            "confidence": round(conf, 4),
            "error_category": category,
            "resume_snippet": row["resume_text"][:250] + "...",
            "job_snippet": row["job_description"][:250] + "...",
        })

    errors_df = pd.DataFrame(errors_list)

    # Save to outputs
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(errors_list, METRICS_DIR / "error_analysis.json")
    errors_df.to_csv(METRICS_DIR / "error_analysis.csv", index=False)
    logger.info(f"Saved qualitative error analysis ({len(errors_list)} examples) to {METRICS_DIR}")

    return errors_df, errors_list


def generate_model_comparison_table(
    distilbert_metrics: Dict[str, any],
) -> pd.DataFrame:
    """
    Consolidate Baseline 1, Baseline 2, and DistilBERT results into a final comparison table.
    """
    baseline_csv = METRICS_DIR / "baseline_metrics.csv"
    if baseline_csv.exists():
        baseline_df = pd.read_csv(baseline_csv)
    else:
        from src.baseline import run_baseline_benchmarks
        baseline_df = run_baseline_benchmarks()

    distil_row = pd.DataFrame([
        {
            "Model": "DistilBERT (Fine-Tuned)",
            "Accuracy": round(distilbert_metrics["accuracy"], 4),
            "Precision": round(distilbert_metrics["precision"], 4),
            "Recall": round(distilbert_metrics["recall"], 4),
            "Macro F1": round(distilbert_metrics["macro_f1"], 4),
            "Weighted F1": round(distilbert_metrics["weighted_f1"], 4),
        }
    ])

    comparison_df = pd.concat([baseline_df, distil_row], ignore_index=True)
    comparison_path = METRICS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Saved final model comparison table to {comparison_path}")

    print("\n=======================================================")
    print("               FINAL MODEL COMPARISON TABLE            ")
    print("=======================================================")
    print(comparison_df.to_string(index=False))
    print("=======================================================\n")

    return comparison_df


def run_full_evaluation(max_test_samples: int = 150) -> pd.DataFrame:
    """Complete evaluation pipeline on test split."""
    set_seed(SEED)

    test_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_path.exists():
        from src.dataset_builder import build_and_save_dataset
        _, _, test_df = build_and_save_dataset()
    else:
        test_df = pd.read_csv(test_path)

    # Subsample stratified test split for fast local evaluation
    if max_test_samples and len(test_df) > max_test_samples:
        n_per_class = max_test_samples // NUM_LABELS
        eval_subset = pd.concat([
            test_df[test_df["label"] == c].sample(n=min(n_per_class, len(test_df[test_df["label"] == c])), random_state=SEED)
            for c in range(NUM_LABELS)
        ]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    else:
        eval_subset = test_df

    logger.info(f"Running full test evaluation on {len(eval_subset)} test pairs...")

    # Run batched DistilBERT inference on test set
    preds, probs = predict_batch_distilbert(eval_subset, max_length=128)

    # Calculate metrics & plots
    distil_metrics = evaluate_transformer(eval_subset, preds, probs)

    # Perform qualitative error analysis
    perform_qualitative_error_analysis(eval_subset, preds, probs, num_examples=12)

    # Generate consolidated comparison table
    comparison_df = generate_model_comparison_table(distil_metrics)

    return comparison_df


if __name__ == "__main__":
    run_full_evaluation()
