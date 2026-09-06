import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import DISTILBERT_MODEL_DIR, HF_HUB_MODEL_ID
from src.utils import get_logger

logger = get_logger("push_to_hub")


def push_model_to_hub(
    model_dir: Path = DISTILBERT_MODEL_DIR,
    repo_id: str = HF_HUB_MODEL_ID,
    token: str = None,
    private: bool = False,
) -> str:
    """
    Publish trained model artifacts to Hugging Face Model Hub.
    """
    load_dotenv()
    hf_token = token or os.getenv("HF_TOKEN")

    if not hf_token:
        logger.warning(
            "No HF_TOKEN found in environment. Attempting to use local huggingface-cli cached credentials..."
        )

    logger.info(f"Target repository ID: {repo_id}")
    logger.info(f"Source model directory: {model_dir}")

    if not model_dir.exists() or not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Model directory not found or incomplete: {model_dir}")

    # Initialize HF API
    api = HfApi()

    # Create repo if not exists
    try:
        create_repo(repo_id=repo_id, token=hf_token, private=private, exist_ok=True)
        logger.info(f"Repository verified: https://huggingface.co/{repo_id}")
    except Exception as e:
        logger.info(f"Repository check: {e}")

    # Load and push tokenizer & model
    logger.info("Uploading tokenizer to Hugging Face Hub...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokenizer.push_to_hub(repo_id=repo_id, token=hf_token)

    logger.info("Uploading model weights to Hugging Face Hub...")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.push_to_hub(repo_id=repo_id, token=hf_token)

    # Upload Model Card README
    model_card_path = model_dir.parent / "README.md"
    if model_card_path.exists():
        logger.info("Uploading Model Card README.md...")
        api.upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=hf_token,
        )

    hub_url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Model successfully deployed to Hugging Face Hub: {hub_url}")
    return hub_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push DistilBERT model to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, default=HF_HUB_MODEL_ID, help="Target Hugging Face repo ID")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API write token")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    push_model_to_hub(repo_id=args.repo_id, token=args.token, private=args.private)
