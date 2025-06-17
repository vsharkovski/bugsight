import argparse
import logging
from io import TextIOWrapper
from pathlib import Path

import pandas as pd

from repstep.logging import LOGGING_FORMAT, PER_INSTANCE_LOGGING_LEVEL
from repstep.repo import checkout_commit, get_repo

SWE_BENCH_COMMON_SPLITS = {
    "dev": "data/dev-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}

MULTIMODAL_EXTENSIONS = set(".js", ".jsx", ".scss", ".frag", ".ts", ".mdx", ".json")

logger = logging.getLogger(__name__)


def load_dataset(dataset_name: str, split_key: str) -> pd.DataFrame:
    logger.info("Loading dataset %s", dataset_name)
    splits = SWE_BENCH_COMMON_SPLITS
    swe_df = pd.read_parquet(f"hf://datasets/{dataset_name}/" + splits[split_key])
    swe_df = swe_df.iloc[:1]
    logger.info("Loaded dataset %s, %s instances", dataset_name, len(swe_df))
    return swe_df


def get_instance_logger(instance_id: str, logs_dir: Path) -> logging.Logger:
    logger_name = f"instance_{instance_id}"
    instance_logger = logging.getLogger(logger_name)

    logger.setLevel(PER_INSTANCE_LOGGING_LEVEL)

    log_filepath = logs_dir / f"{logger_name}.log"
    handler = logging.FileHandler(log_filepath, mode="a")
    instance_logger.addHandler(handler)

    formatter = logging.Formatter(LOGGING_FORMAT)
    handler.setFormatter(formatter)

    # Don't propagate to the root logger
    instance_logger.propagate = False

    return instance_logger


def get_files_filtered(repo_dir: Path, filter_multimodal: bool) -> list[Path]:
    file_paths = repo_dir.glob("**/*")

    # Filter out test files
    file_paths = [path for path in file_paths if not any("test" in str(path))]

    if filter_multimodal:
        file_paths = [
            path for path in file_paths if path.suffix in MULTIMODAL_EXTENSIONS
        ]

    # Get absolute paths
    file_paths = [path.absolute() for path in file_paths]

    return file_paths


def get_file_to_contents(
    file_paths: list[Path], instance_logger: logging.Logger
) -> dict[Path, str]:
    file_to_contents = {}
    for file_path in file_paths:
        try:
            with file_path.open() as file:
                file_to_contents[file_path] = file.read()
        except Exception as e:
            instance_logger.error(
                "Failed to read file %s: %s", file_path, e, exc_info=e
            )

    return file_to_contents


def retrieve_instance(
    instance: pd.Series,
    logs_dir: Path,
    testbed_dir: Path,
    filter_multimodal_files: bool,
    persist_dir: Path,
    output_file: TextIOWrapper,
):
    instance_id = instance["instance_id"]
    instance_logger = get_instance_logger(instance_id, logs_dir)
    logger.info("Starting retrieval for instance %s", instance_id)

    repo_id = instance["repo"]
    repo_dir = get_repo(repo_id, testbed_dir, instance_logger)
    commit_id = instance["base_commit"]
    checkout_commit(repo_dir, commit_id, instance_logger)

    file_paths = get_files_filtered(repo_dir, filter_multimodal_files)
    instance_logger.info("Found %s relevant files", len(file_paths))

    file_to_contents = get_file_to_contents(file_paths, instance_logger)

    instance_logger.info("Starting retrieval process")
    instance_logger.info("Using persist directory: %s", persist_dir)


def retrieve_swe(args: argparse.Namespace):
    logger.info("Initializing directories")
    results_dir = Path(args.results_dir)
    logs_dir = results_dir / "logs"

    logs_per_instance_dir = logs_dir / "per_instance"
    logs_per_instance_dir.mkdir(exist_ok=True)

    testbed_dir = results_dir / "testbed"
    testbed_dir.mkdir(exist_ok=True)

    persist_dir = Path(args.embedding_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    swe_df = load_dataset(args.dataset, args.split)

    # Multithreading is not implemented yet.
    num_threads = 1
    logger.info("Running with %s thread(s)", num_threads)

    output_filepath: Path = results_dir / args.output_file
    output_file = output_filepath.open(mode="w")

    for _, instance in swe_df.iterrows():
        retrieve_instance(
            instance,
            logs_per_instance_dir,
            testbed_dir,
            args.filter_multimodal,
            persist_dir,
            output_file,
        )

    output_file.close()
    logger.info("Completed all retrievals")
