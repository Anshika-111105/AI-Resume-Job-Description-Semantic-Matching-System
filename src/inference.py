import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    DISTILBERT_MODEL_DIR,
    ID2LABEL,
    LABEL2ID,
    MAX_LENGTH,
    MODELS_DIR,
    NUM_LABELS,
)
from src.preprocessing import clean_text, prepare_text_pair
from src.skill_extractor import analyze_skill_gap
from src.utils import get_device, get_logger

logger = get_logger("inference")


class ResumeJobMatcher:
    """
    Inference engine for Resume-Job Description Semantic Matching.
    Supports DistilBERT Transformer with seamless fallback to Logistic Regression baseline.
    """
    def __init__(
        self,
        model_dir: Union[str, Path] = DISTILBERT_MODEL_DIR,
        device: Optional[torch.device] = None,
    ):
        self.model_dir = Path(model_dir)
        self.device = device or get_device()
        self.model = None
        self.tokenizer = None
        self.baseline_pipeline = None
        self.model_type = "none"

        self._load_model()

    def _load_model(self) -> None:
        """Load Transformer model or fall back to saved Logistic Regression baseline."""
        # 1. Try loading fine-tuned DistilBERT
        if (self.model_dir / "config.json").exists() and (
            (self.model_dir / "model.safetensors").exists()
            or (self.model_dir / "pytorch_model.bin").exists()
        ):
            try:
                logger.info(f"Loading fine-tuned DistilBERT from {self.model_dir}...")
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
                self.model.to(self.device)
                self.model.eval()
                self.model_type = "distilbert"
                logger.info("DistilBERT loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Could not load DistilBERT model from {self.model_dir}: {e}")

        # 2. Try loading baseline logistic regression model
        baseline_path = MODELS_DIR / "logistic_regression_baseline.joblib"
        if baseline_path.exists():
            try:
                logger.info(f"Loading baseline model from {baseline_path}...")
                self.baseline_pipeline = joblib.load(baseline_path)
                self.model_type = "logistic_regression"
                logger.info("Baseline Logistic Regression model loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Could not load baseline model: {e}")

        # 3. If neither checkpoint exists, initialize pretrained distilbert-base-uncased directly
        try:
            logger.info("Initializing pre-trained distilbert-base-uncased for zero-shot evaluation...")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=NUM_LABELS,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
            self.model.to(self.device)
            self.model.eval()
            self.model_type = "distilbert_pretrained"
        except Exception as e:
            logger.error(f"Failed to load any model: {e}")

    def predict(self, resume_text: str, job_description: str,) -> Dict[str, any]:
        """
        Run end-to-end semantic match prediction on candidate resume and job description.
        """
        start_time = time.perf_counter()

        clean_resume, clean_jd = prepare_text_pair(resume_text, job_description)

        if not clean_resume:
            raise ValueError("Resume text is empty or could not be extracted.")
        if not clean_jd:
            raise ValueError("Job description is empty.")

        # Run model inference
        if self.model_type.startswith("distilbert") and self.model is not None:
            encoded = self.tokenizer(
                text=clean_resume,
                text_pair=clean_jd,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        elif self.model_type == "logistic_regression" and self.baseline_pipeline is not None:
            pair_text = f"RESUME: {clean_resume} \n\n JOB_DESCRIPTION: {clean_jd}"
            probs = self.baseline_pipeline.predict_proba([pair_text])[0]
        else:
            # Fallback uniform probability distribution
            probs = np.array([0.33, 0.33, 0.34])

        pred_class_idx = int(np.argmax(probs))
        label_name = ID2LABEL.get(pred_class_idx, "Unknown")
        confidence = float(probs[pred_class_idx])

        # Compute continuous Model Match Score (0% to 100%)
        # Weighted expectation: 0.0 * P(No Fit) + 0.5 * P(Potential Fit) + 1.0 * P(Good Fit)
        composite_score = float(0.0 * probs[0] + 0.5 * probs[1] + 1.0 * probs[2])
        score_percentage = f"{int(round(composite_score * 100))}%"

        # Probability dictionary
        prob_dict = {
            ID2LABEL[0]: round(float(probs[0]), 4),
            ID2LABEL[1]: round(float(probs[1]), 4),
            ID2LABEL[2]: round(float(probs[2]), 4),
        }

        # Skill gap analysis
        skill_analysis = analyze_skill_gap(clean_resume, clean_jd)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        result = {
            "label": label_name,
            "label_id": pred_class_idx,
            "score": round(composite_score, 4),
            "score_percentage": score_percentage,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "model_type": self.model_type,
            "latency_ms": latency_ms,
            "skill_analysis": skill_analysis,
        }
        return result


# Global singleton instance for Streamlit and fast reuse
_MATCHER_INSTANCE = None


def get_matcher_instance(model_dir: Path = DISTILBERT_MODEL_DIR) -> ResumeJobMatcher:
    global _MATCHER_INSTANCE
    if _MATCHER_INSTANCE is None:
        _MATCHER_INSTANCE = ResumeJobMatcher(model_dir=model_dir)
    return _MATCHER_INSTANCE


def predict_match(resume_text: str,job_description: str,model_dir: Path = DISTILBERT_MODEL_DIR,) -> Dict[str, any]:
    """
    Convenience functional API for single pair prediction.
    """
    matcher = get_matcher_instance(model_dir=model_dir)
    return matcher.predict(resume_text, job_description)


if __name__ == "__main__":
    test_resume = """
    Senior Data Scientist with 6 years of experience.
    Proficient in Python, SQL, Scikit-learn, PyTorch, Pandas, and AWS.
    Experience deploying predictive models and designing A/B tests.
    """
    test_jd = """
    Looking for a Senior Machine Learning Engineer with strong Python, SQL, PyTorch,
    Docker, and AWS cloud deployment experience.
    """

    res = predict_match(test_resume, test_jd)
    print("--- Inference Prediction Demo ---")
    print("Predicted Label:   ", res["label"])
    print("Model Match Score: ", res["score_percentage"])
    print("Probabilities:     ", res["probabilities"])
    print("Matching Skills:   ", res["skill_analysis"]["matching_skills"])
    print("Missing Skills:    ", res["skill_analysis"]["missing_skills"])
    print("Latency:           ", res["latency_ms"], "ms")
