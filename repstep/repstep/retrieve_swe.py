import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from repstep.data import load_dataset_from_disk
from repstep.embeddings import FilterOptions, retrieve
from repstep.logging import PER_INSTANCE_LOGGING_LEVEL
from repstep.repo import checkout_commit, get_repo

MULTIMODAL_EXTENSIONS = set([".js", ".jsx", ".scss", ".frag", ".ts", ".mdx", ".json"])

RETRIEVAL_ISSUE_PROMPT = (
    "Issue: ```\n{problem_statement}\n```\n\n"
    "Given the above issue, find the code which needs to be edited to solve the issue."
)
RETRIEVAL_TRANSCRIPTION_PROMPT = (
    "Bug report image transcription: ```\n{transcription}\n```\n\n"
    "Given the above bug report image transcription, find the code which needs to be edited to solve the issue."
)

logger = logging.getLogger(__name__)


def get_instance_logger(instance_id: str) -> logging.Logger:
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

    # Filter out test files, i.e. files containing "test" in their path.
    # Because all paths by default start with repo_dir,
    # if this prefix contains "test" in it, all paths will as well.
    # Therefore relativize paths in relation to this prefix before checking.
    file_paths = [
        path for path in file_paths if "test" not in str(path.relative_to(repo_dir))
    ]

    if filter_multimodal:
        file_paths = [
            path for path in file_paths if path.suffix in MULTIMODAL_EXTENSIONS
        ]

    # Get absolute paths
    file_paths = [path.absolute() for path in file_paths]

    return file_paths


def retrieve_instance(
    instance: pd.Series,
    retrieval_field: str,
    testbed_dir: Path,
    filter_multimodal_files: bool,
    filter_options: Optional[FilterOptions],
    embeddings_dir: Path,
    embedding_model_name: str,
    just_create_retrieval_index: bool,
    retrieve_count: int,
    retrieve_entire_files: bool,
) -> dict[str, Any]:
    instance_id = instance["instance_id"]
    instance_logger = get_instance_logger(instance_id)
    logger.info("Starting retrieval for instance: %s", instance_id)

    if retrieval_field == "problem_statement":
        retrieval_field_value = instance["problem_statement"]
        retrieval_prompt = RETRIEVAL_ISSUE_PROMPT.format(
            problem_statement=retrieval_field_value
        )
    elif retrieval_field == "transcription":
        retrieval_field_value = instance["transcription"]
        retrieval_prompt = RETRIEVAL_TRANSCRIPTION_PROMPT.format(
            transcription=retrieval_field_value
        )
    else:
        raise Exception(f"Unsupported retrieval field: {retrieval_field}")

    repo_id = instance["repo"]
    repo_dir = get_repo(repo_id, testbed_dir, instance_logger)
    commit_id = instance["base_commit"]
    checkout_commit(repo_dir, commit_id, instance_logger)

    file_paths = get_files_filtered(repo_dir, filter_multimodal_files)
    instance_logger.info("Found %s relevant files", len(file_paths))

    instance_logger.info(
        "Starting retrieval process, retrieving for field %s", retrieval_field
    )
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

    retrieved_file_paths_strs = [str(path) for path in retrieved_file_paths]
    result = {
        "instance_id": instance_id,
        "retrieval_field": retrieval_field,
        "retrieval_field_value": retrieval_field_value,
        "found_files": retrieved_file_paths_strs,
        "file_contents": retrieved_file_contents,
    }

    if "image_assets" in instance:
        image_assets_obj = json.loads(instance["image_assets"])
        result["image_assets"] = image_assets_obj["problem_statement"]

    return result


def retrieve_swe(args: argparse.Namespace):
    logger.info("Initializing directories")
    testbed_dir_str: str = args.testbed_dir
    testbed_dir = Path(testbed_dir_str)
    testbed_dir.mkdir(parents=True, exist_ok=True)

    embeddings_dir_str: str = args.embedding_dir
    embeddings_dir = Path(embeddings_dir_str)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    retrieval_field: str = args.retrieval_field
    logger.info("Using retrieval field: %s", retrieval_field)

    instances_filepath_str: str = args.instances_file
    instances_filepath = Path(instances_filepath_str)
    logger.info("Loading instances file: %s", instances_filepath)
    swe_df = load_dataset_from_disk(instances_filepath)

    # Multithreading is not implemented yet.
    num_threads = 1
    logger.info("Running with %s thread(s)", num_threads)

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

    output_filepath_str: str = args.output_file
    output_filepath = Path(output_filepath_str)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    with output_filepath.open(mode="w") as output_file:
        for _, instance in swe_df.iterrows():
            result = retrieve_instance(
                instance,
                retrieval_field,
                testbed_dir,
                args.filter_multimodal,
                filter_options,
                embeddings_dir,
                embedding_model_name,
                just_create_retrieval_index,
                retrieve_count,
                retrieve_entire_files,
            )

            json_line = json.dumps(result)
            output_file.write(json_line)
            output_file.write("\n")
            output_file.flush()

    logger.info("Completed all retrievals")
