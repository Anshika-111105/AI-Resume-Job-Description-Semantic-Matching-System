"""
General Utility Functions
AI Resume–Job Description Semantic Matching System
"""

import os
import random
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import SEED, FIGURES_DIR


def set_seed(seed: int = SEED) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Detect and return the primary compute device (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_logger(name: str = "resume_matcher") -> logging.Logger:
    """Create a structured console and file logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save dictionary to a formatted JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file into a Python dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_f1s: list,
    save_dir: Path = FIGURES_DIR,
) -> None:
    """
    Generate and save training loss, validation loss, and F1 score curves.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training Loss
    if train_losses:
        plt.figure(figsize=(8, 5))
        epochs_train = range(1, len(train_losses) + 1)
        plt.plot(epochs_train, train_losses, "b-o", label="Training Loss", linewidth=2, markersize=6)
        plt.title("Training Loss vs Steps/Epochs", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Logging Interval / Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(save_dir / "training_loss.png", dpi=300)
        plt.close()

    # 2. Validation Loss
    if val_losses:
        plt.figure(figsize=(8, 5))
        epochs_val = range(1, len(val_losses) + 1)
        plt.plot(epochs_val, val_losses, "r-s", label="Validation Loss", linewidth=2, markersize=6)
        plt.title("Validation Loss vs Epoch", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(save_dir / "validation_loss.png", dpi=300)
        plt.close()

    # 3. Validation F1
    if val_f1s:
        plt.figure(figsize=(8, 5))
        epochs_f1 = range(1, len(val_f1s) + 1)
        plt.plot(epochs_f1, val_f1s, "g-^", label="Validation Macro F1", linewidth=2, markersize=6)
        plt.title("Validation Macro F1 vs Epoch", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Macro F1", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(save_dir / "f1_curve.png", dpi=300)
        plt.close()
