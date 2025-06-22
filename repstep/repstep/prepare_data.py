import argparse
import logging
from pathlib import Path

import pandas as pd

from repstep.data import DEFAULT_DATASET_NAME, load_dataset_from_huggingface

logger = logging.getLogger(__name__)

"""
    Subset 1: text has steps to reproduce --> remove text, try to produce reproduction steps. You have original as ground truth to compare to
    Subset 2: text doesn't have steps to reproduce --> 2.1 keep text+image and try to produce reproduction steps. This just tells us if our adaptation to produce reproduction steps works -- so baby steps. 2.2 remove text and try to produce reproduction steps. This tells us if we can really create reproduction steps from just an image.

In both these subsets, for now, we should choose issues that the model was able to solve successfully



"""

INSTANCE_IDS_WITH_S2R = []
INSTANCE_IDS_NO_S2R = []


def set_subset_column(swe_df: pd.DataFrame):
    swe_df["subset"] = "none"

    swe_df_with_s2r_mask = swe_df["instance_id"].isin(INSTANCE_IDS_WITH_S2R)
    swe_df.loc[swe_df_with_s2r_mask, "subset"] = "with_s2r"

    swe_df_no_s2r_mask = swe_df["instance_id"].isin(INSTANCE_IDS_NO_S2R)
    swe_df.loc[swe_df_no_s2r_mask, "subset"] = "no_s2r"


def prepare_data(results_dir: Path, args: argparse.Namespace):
    swe_df = load_dataset_from_huggingface(args.dataset, args.split)

    set_subset_column(swe_df)

    # Save data
    data_dir = results_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    save_path = data_dir / DEFAULT_DATASET_NAME
    swe_df.to_parquet(save_path)
