import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from repstep.backends import BaseGenerator, get_generator_from_model_name
from repstep.data import load_dataset_from_disk

TRANSCRIPTION_PROMPT = "You will be given images from a bug report. Nothing is known about the bug report besides the provided images. Your task is to transcribe the images into text. Your transcription will be used to locate relevant files in the code base, in order to aid developers with fixing the bug. Please transcribe the images, mentioning any relevant elements. Because your transcription will be used for a search, use simple concise language."

logger = logging.getLogger(__name__)


def transcribe_instance(
    instance: pd.Series,
    generator: BaseGenerator,
) -> str:
    image_assets_obj = json.loads(instance["image_assets"])
    image_urls: list[str] = image_assets_obj["problem_statement"]
    completions = generator.generate(TRANSCRIPTION_PROMPT, image_urls)
    transcription = completions[0]
    logger.info(
        "Transcription for instance %s: %s", instance["instance_id"], transcription
    )
    return transcription


def transcribe(args: argparse.Namespace):
    instances_filepath_str: str = args.instances_file
    instances_filepath = Path(instances_filepath_str)
    logger.info("Loading instances file: %s", instances_filepath)
    swe_df = load_dataset_from_disk(instances_filepath)

    transcription_model_name: str = args.model
    generator = get_generator_from_model_name(transcription_model_name)

    logger.info(
        "Starting transcription of %s instances with model %s",
        len(swe_df),
        transcription_model_name,
    )

    transcriptions_list: list[str] = []

    for _, instance in swe_df.iterrows():
        transcription = transcribe_instance(instance, generator)
        transcriptions_list.append(transcription)

    logger.info("Completed all transcriptions, saving to output file")

    transcriptions_series = pd.Series(data=transcriptions_list)
    swe_df["transcription"] = transcriptions_series

    rows_to_print = ["instance_id", "subset", "image_assets", "transcription"]
    logger.info("Data:\n%s", swe_df[rows_to_print].to_string())

    output_filepath_str: str = args.output_file
    output_filepath = Path(output_filepath_str)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    swe_df.to_parquet(output_filepath)
