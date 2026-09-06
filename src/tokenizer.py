from typing import Dict, List, Optional, Tuple, Union
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.config import MODEL_NAME, MAX_LENGTH
from src.utils import get_logger

logger = get_logger("tokenizer")


def get_tokenizer(model_name: str = MODEL_NAME) -> PreTrainedTokenizerFast:
    """
    Load and return the Hugging Face pre-trained tokenizer for DistilBERT.
    """
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def tokenize_pair(
    resume_text: str,
    job_description: str,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    max_length: int = MAX_LENGTH,
    padding: Union[bool, str] = "max_length",
    truncation: Union[bool, str] = True,
    return_tensors: Optional[str] = None,
) -> Dict[str, Union[List[int], torch.Tensor]]:
    """
    Tokenize a single candidate resume and job description pair using DistilBERT special tokens.
    Format: [CLS] resume_text [SEP] job_description [SEP]
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    encoded = tokenizer(
        text=resume_text,
        text_pair=job_description,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )
    return encoded


def explain_tokenization(
    sample_resume: str,
    sample_jd: str,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
) -> Dict[str, any]:
    """
    Educational inspection helper: Breaks down tokenization components.
    Explains [CLS], [SEP], Input IDs, Attention Mask, and Truncation.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    encoded = tokenizer(
        text=sample_resume,
        text_pair=sample_jd,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors=None,
    )

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

    explanation = {
        "cls_token": tokenizer.cls_token,
        "cls_token_id": tokenizer.cls_token_id,
        "sep_token": tokenizer.sep_token,
        "sep_token_id": tokenizer.sep_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "total_tokens": len(encoded["input_ids"]),
        "tokens_preview": tokens[:20],
        "input_ids_preview": encoded["input_ids"][:20],
        "attention_mask_preview": encoded["attention_mask"][:20],
    }
    return explanation
