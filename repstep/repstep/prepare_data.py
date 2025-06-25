import argparse
import logging
from pathlib import Path

import pandas as pd

from repstep.data import load_dataset_from_huggingface

logger = logging.getLogger(__name__)

"""
Subset 1: text has steps to reproduce --> remove text, try to produce reproduction steps. You have original as ground truth to compare to
Subset 2: text doesn't have steps to reproduce --> 2.1 keep text+image and try to produce reproduction steps. This just tells us if our adaptation to produce reproduction steps works -- so baby steps. 2.2 remove text and try to produce reproduction steps. This tells us if we can really create reproduction steps from just an image.

In both these subsets, for now, we should choose issues that the model was able to solve successfully
"""

INSTANCE_IDS_WITH_S2R: list[str] = ["chartjs__Chart.js-10301"]
INSTANCE_IDS_NO_S2R: list[str] = []


def set_subset_column(swe_df: pd.DataFrame):
    swe_df["subset"] = "none"

    swe_df_with_s2r_mask = swe_df["instance_id"].isin(INSTANCE_IDS_WITH_S2R)
    swe_df.loc[swe_df_with_s2r_mask, "subset"] = "with_s2r"

    swe_df_no_s2r_mask = swe_df["instance_id"].isin(INSTANCE_IDS_NO_S2R)
    swe_df.loc[swe_df_no_s2r_mask, "subset"] = "no_s2r"


def prepare_data(args: argparse.Namespace):
    swe_df = load_dataset_from_huggingface(args.dataset, args.split)

    # Set subsets we are interested in, and remove instances not part of a subset
    set_subset_column(swe_df)
    swe_df = swe_df[swe_df["subset"].ne("none")]

    rows_to_print = ["instance_id", "subset"]
    logger.info("Data:\n%s", swe_df[rows_to_print].to_string())

    # Save data
    output_filepath_str: str = args.output_file
    output_filepath = Path(output_filepath_str)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving %s rows to %s", len(swe_df), output_filepath)
    swe_df.to_parquet(output_filepath)
