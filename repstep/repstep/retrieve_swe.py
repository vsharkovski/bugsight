import argparse
import json
import logging
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from repstep.data import load_dataset_from_disk
from repstep.embeddings import FilterOptions, retrieve
from repstep.logging import PER_INSTANCE_LOGGING_LEVEL
from repstep.repo import checkout_commit, get_repo

MULTIMODAL_EXTENSIONS = set([".js", ".jsx", ".scss", ".frag", ".ts", ".mdx", ".json"])

RETRIEVAL_INSTRUCTION = (
    "Find code the code which need to be edited to solve the above issue."
)

logger = logging.getLogger(__name__)


def get_instance_logger(instance_id: str, logs_dir: Path) -> logging.Logger:
    logger_name = f"instance_{instance_id}"
    instance_logger = logging.getLogger(logger_name)

    logger.setLevel(PER_INSTANCE_LOGGING_LEVEL)

    # log_filepath = logs_dir / f"{logger_name}.log"
    # handler = logging.FileHandler(log_filepath, mode="a")
    # instance_logger.addHandler(handler)

    # formatter = logging.Formatter(LOGGING_FORMAT)
    # handler.setFormatter(formatter)

    # Don't propagate to the root logger
    # instance_logger.propagate = False

    return instance_logger


def get_files_filtered(repo_dir: Path, filter_multimodal: bool) -> list[Path]:
    file_paths = list(repo_dir.glob("**/*"))

    # Filter out test files
    file_paths = [path for path in file_paths if "test" not in str(path)]

    if filter_multimodal:
        file_paths = [
            path for path in file_paths if path.suffix in MULTIMODAL_EXTENSIONS
        ]

    # Get absolute paths
    file_paths = [path.absolute() for path in file_paths]

    return file_paths


def retrieve_instance(
    instance: pd.Series,
    logs_dir: Path,
    testbed_dir: Path,
    filter_multimodal_files: bool,
    filter_options: Optional[FilterOptions],
    embeddings_dir: Path,
    embedding_model_name: str,
    just_create_retrieval_index: bool,
    retrieve_count: int,
    retrieve_entire_files: bool,
    output_file: TextIOWrapper,
) -> dict[str, Any]:
    instance_id = instance["instance_id"]
    instance_logger = get_instance_logger(instance_id, logs_dir)
    logger.info("Starting retrieval for instance %s", instance_id)

    repo_id = instance["repo"]
    repo_dir = get_repo(repo_id, testbed_dir, instance_logger)
    commit_id = instance["base_commit"]
    checkout_commit(repo_dir, commit_id, instance_logger)

    file_paths = get_files_filtered(repo_dir, filter_multimodal_files)
    instance_logger.info("Found %s relevant files", len(file_paths))

    instance_logger.info("Starting retrieval process")
    original_prompt = instance["problem_statement"]
    retrieval_prompt = f"{original_prompt}\n\n{RETRIEVAL_INSTRUCTION}"
    retrieved_file_paths, retrieved_file_contents = retrieve(
        embeddings_dir,
        retrieval_prompt,
        file_paths,
        filter_options,
        embedding_model_name,
        just_create_retrieval_index,
        retrieve_count,
        retrieve_entire_files,
        instance_logger,
    )
    instance_logger.info("Retrieved %s files", len(retrieved_file_paths))

    result = {
        "instance_id": instance_id,
        "problem_description": original_prompt,
        "found_files": retrieved_file_paths,
        "file_contents": retrieved_file_contents,
    }

    if "image_assets" in instance:
        image_assets_obj = json.loads(instance["image_assets"])
        result["image_assets"] = image_assets_obj["problem_statement"]

    json_line = json.dumps(result)
    output_file.write(json_line)
    output_file.write("\n")
    output_file.flush()
    instance_logger.info("Successfully wrote results to output file")
    return result


def retrieve_swe(results_dir: Path, logs_dir: Path, args: argparse.Namespace):
    logger.info("Initializing directories")
    logs_per_instance_dir = logs_dir / "per_instance"
    logs_per_instance_dir.mkdir(exist_ok=True)

    testbed_dir = results_dir / "testbed"
    testbed_dir.mkdir(exist_ok=True)

    embeddings_dir_str: str = args.embedding_dir
    embeddings_dir = Path(embeddings_dir_str)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    swe_df = load_dataset_from_disk()
    swe_df = swe_df.iloc[:1]

    # Multithreading is not implemented yet.
    num_threads = 1
    logger.info("Running with %s thread(s)", num_threads)

    output_file_name: str = args.output_file
    output_filepath = results_dir / output_file_name
    output_file = output_filepath.open(mode="w")

    filter_options_values = [args.filter_model, args.filter_count]
    if any(v is None for v in filter_options_values):
        if not all(v is None for v in filter_options_values):
            raise Exception(
                "Filter options (filter model, filter count) must either all be provided or none."
            )
        filter_options = None
    else:
        filter_options = FilterOptions(
            filter_model_name=args.filter_model, filter_count=args.filter_count
        )

    embedding_model_name: str = args.embedding_model
    just_create_retrieval_index: bool = args.just_create_index
    retrieve_count: int = args.retrieve_count
    retrieve_entire_files: bool = args.entire_file

    for _, instance in swe_df.iterrows():
        retrieve_instance(
            instance,
            logs_per_instance_dir,
            testbed_dir,
            args.filter_multimodal,
            filter_options,
            embeddings_dir,
            embedding_model_name,
            just_create_retrieval_index,
            retrieve_count,
            retrieve_entire_files,
            output_file,
        )

    output_file.close()
    logger.info("Completed all retrievals")
