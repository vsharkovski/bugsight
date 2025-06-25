import logging
from pathlib import Path

import pandas as pd

SWE_BENCH_COMMON_SPLITS = {
    "dev": "data/dev-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}

logger = logging.getLogger(__name__)


def load_dataset_from_huggingface(dataset_name: str, split_key: str) -> pd.DataFrame:
    logger.info(
        "Loading dataset from HuggingFace with name %s, split %s",
        dataset_name,
        split_key,
    )
    splits = SWE_BENCH_COMMON_SPLITS
    swe_df = pd.read_parquet(f"hf://datasets/{dataset_name}/" + splits[split_key])
    logger.info(
        "Loaded dataset %s, split %s, %s instances",
        dataset_name,
        split_key,
        len(swe_df),
    )
    return swe_df


def load_dataset_from_disk(data_filepath: Path) -> pd.DataFrame:
    result = pd.read_parquet(data_filepath)
    return result
