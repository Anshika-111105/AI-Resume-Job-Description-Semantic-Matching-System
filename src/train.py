import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import (
    MODEL_NAME,
    MAX_LENGTH,
    LEARNING_RATE,
    BATCH_SIZE,
    EVAL_BATCH_SIZE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    SEED,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    DISTILBERT_MODEL_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    METRICS_DIR,
)
from src.tokenizer import get_tokenizer
from src.utils import get_logger, get_device, set_seed, save_json, plot_training_curves

logger = get_logger("train_distilbert")


class ResumeJobDataset(Dataset):
    """
    PyTorch Dataset for Resume and Job Description pairs.
    """
    def __init__(
        self,
        resume_texts: List[str],
        job_descriptions: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = MAX_LENGTH,
    ):
        self.resume_texts = resume_texts
        self.job_descriptions = job_descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        resume = str(self.resume_texts[idx])
        jd = str(self.job_descriptions[idx])
        label = int(self.labels[idx])

        encoded = self.tokenizer(
            text=resume,
            text_pair=jd,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute classification metrics for Hugging Face Trainer evaluation.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
    }


class MetricsHistoryCallback(TrainerCallback):
    """
    Custom callback to record training and validation loss and F1 history per epoch.
    """
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.val_f1s = []
        self.val_accuracies = []
        self.epochs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Check if validation log
        if "eval_loss" in logs:
            self.val_losses.append(logs.get("eval_loss"))
            self.val_f1s.append(logs.get("eval_macro_f1", 0.0))
            self.val_accuracies.append(logs.get("eval_accuracy", 0.0))
            current_epoch = int(round(logs.get("epoch", len(self.val_losses))))
            self.epochs.append(current_epoch)

        # Record training loss
        if "loss" in logs and "eval_loss" not in logs:
            self.train_losses.append(logs.get("loss"))


def train_distilbert(
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_length: int = 128,
    num_epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = LEARNING_RATE,
    output_dir: Path = DISTILBERT_MODEL_DIR,
    seed: int = SEED,
) -> Tuple[Trainer, Dict[str, any]]:
    """
    Fine-tune DistilBERT on candidate-job description semantic matching.
    """
    set_seed(seed)
    device = get_device()
    logger.info(f"Using compute device: {device}")

    # Load data if not provided
    if train_df is None or val_df is None:
        train_path = PROCESSED_DATA_DIR / "train.csv"
        val_path = PROCESSED_DATA_DIR / "val.csv"
        if not train_path.exists() or not val_path.exists():
            from src.dataset_builder import build_and_save_dataset
            train_df, val_df, _ = build_and_save_dataset()
        else:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)

    # Subsample if requested (e.g. for rapid CPU verification)
    if max_train_samples and len(train_df) > max_train_samples:
        n_per_class = max_train_samples // NUM_LABELS
        logger.info(f"Subsampling train set to {max_train_samples} samples ({n_per_class} per class).")
        train_df = pd.concat([
            train_df[train_df["label"] == c].sample(n=min(n_per_class, len(train_df[train_df["label"] == c])), random_state=seed)
            for c in range(NUM_LABELS)
        ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if max_val_samples and len(val_df) > max_val_samples:
        n_per_class = max_val_samples // NUM_LABELS
        logger.info(f"Subsampling val set to {max_val_samples} samples ({n_per_class} per class).")
        val_df = pd.concat([
            val_df[val_df["label"] == c].sample(n=min(n_per_class, len(val_df[val_df["label"] == c])), random_state=seed)
            for c in range(NUM_LABELS)
        ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info(f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples (max_length={max_length}).")

    tokenizer = get_tokenizer(MODEL_NAME)

    train_dataset = ResumeJobDataset(
        resume_texts=train_df["resume_text"].tolist(),
        job_descriptions=train_df["job_description"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    val_dataset = ResumeJobDataset(
        resume_texts=val_df["resume_text"].tolist(),
        job_descriptions=val_df["job_description"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    logger.info(f"Initializing pre-trained {MODEL_NAME} for 3-class sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=num_epochs,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=5,
        save_total_limit=1,
        seed=seed,
        report_to="none",
        use_cpu=(device.type == "cpu"),
    )

    history_callback = MetricsHistoryCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[history_callback],
    )

    logger.info("Starting DistilBERT fine-tuning...")
    train_result = trainer.train()

    logger.info(f"Training completed. Global steps: {train_result.global_step}, Train Loss: {train_result.training_loss:.4f}")

    # Save best model and tokenizer
    logger.info(f"Saving fine-tuned model and tokenizer to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Evaluate on validation set
    val_metrics = trainer.evaluate()
    logger.info(f"Final Validation Metrics: {val_metrics}")

    # Generate training curves
    train_losses = history_callback.train_losses if history_callback.train_losses else [train_result.training_loss]
    val_losses = history_callback.val_losses if history_callback.val_losses else [val_metrics.get("eval_loss", 0.0)]
    val_f1s = history_callback.val_f1s if history_callback.val_f1s else [val_metrics.get("eval_macro_f1", 0.0)]

    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        val_f1s=val_f1s,
        save_dir=FIGURES_DIR,
    )
    logger.info(f"Saved training curves to {FIGURES_DIR}")

    # Save training history JSON
    history = {
        "model_name": MODEL_NAME,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "train_loss_final": float(train_result.training_loss),
        "val_loss_final": float(val_metrics.get("eval_loss", 0.0)),
        "val_macro_f1_final": float(val_metrics.get("eval_macro_f1", 0.0)),
        "val_accuracy_final": float(val_metrics.get("eval_accuracy", 0.0)),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "val_metrics": val_metrics,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(history, METRICS_DIR / "training_history.json")

    return trainer, history


if __name__ == "__main__":
    # Train representative fine-tuning run locally on CPU
    trainer, history = train_distilbert(
        max_train_samples=90,
        max_val_samples=30,
        max_length=128,
        num_epochs=2,
        batch_size=15,
    )
    print("\nDistilBERT Fine-Tuning Completed Successfully!")
    print(f"Validation Macro F1: {history['val_macro_f1_final']:.4f}")
    print(f"Validation Accuracy: {history['val_accuracy_final']:.4f}")
