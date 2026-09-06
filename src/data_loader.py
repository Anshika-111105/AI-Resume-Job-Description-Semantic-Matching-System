from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
from datasets import load_dataset, Dataset

from src.config import HF_DATASET_NAME, RAW_DATA_DIR
from src.utils import get_logger

logger = get_logger("data_loader")


def load_raw_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Load raw resumes dataset from Hugging Face Hub (michaelozon/candidate-matching-synthetic).
    """
    raw_cache_path = RAW_DATA_DIR / "raw_resumes.csv"

    if raw_cache_path.exists() and not force_download:
        logger.info(f"Loading raw resumes from local cache: {raw_cache_path}")
        df = pd.read_csv(raw_cache_path)
        # Convert string representations of lists back to Python lists if needed
        import ast
        for col in ["skills", "experience_bullets"]:
            if col in df.columns and isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x)
        return df

    logger.info(f"Fetching dataset from Hugging Face Hub: {HF_DATASET_NAME}")
    try:
        ds = load_dataset(HF_DATASET_NAME, split="resumes")
        df = pd.DataFrame(ds)
        logger.info(f"Dataset successfully loaded. Shape: {df.shape}")

        # Save to local raw cache
        raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_cache_path, index=False)
        logger.info(f"Cached raw dataset to {raw_cache_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset from HF Hub: {e}")
        raise e


def inspect_dataset_schema(df: pd.DataFrame) -> Dict[str, any]:
    """
    Programmatically inspect the dataset schema, data types, missing values,
    and distributions without making arbitrary assumptions.
    """
    schema_info = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "unique_roles_count": int(df["role"].nunique()) if "role" in df else 0,
        "unique_seniorities": df["seniority"].value_counts().to_dict() if "seniority" in df else {},
        "unique_industries": df["industry"].value_counts().to_dict() if "industry" in df else {},
    }
    return schema_info


if __name__ == "__main__":
    df = load_raw_dataset()
    schema = inspect_dataset_schema(df)
    print("Schema Inspection Results:")
    for k, v in schema.items():
        print(f"  {k}: {v}")
