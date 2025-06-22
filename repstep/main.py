import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from repstep.logging import LOGGING_FORMAT, ROOT_LOGGING_LEVEL
from repstep.prepare_data import prepare_data
from repstep.retrieve_swe import retrieve_swe


def build_get_steps_parser(subparser: argparse.ArgumentParser):
    pass


def build_retrieve_swe_parser(subparser: argparse.ArgumentParser):
    subparser.add_argument(
        "--output_file",
        type=str,
        default="retrievals.jsonl",
        help="Output file name for the results",
    )

    subparser.add_argument(
        "--embedding_dir",
        type=str,
        default="embeddings",
        help="Directory for embeddings",
    )
    subparser.add_argument(
        "--embedding_model", type=str, default="text-embedding-3-small"
    )

    subparser.add_argument(
        "--filter_multimodal",
        action="store_true",
        help="Filter out files for multimodal SWE-bench",
    )
    subparser.add_argument(
        "--filter_model",
        type=str,
        default=None,
        help="Model to use for initial filter. If none is specified, then do not perform initial filtering",
    )
    subparser.add_argument(
        "--filter_count",
        type=int,
        default=300,
        help="Number of snippets to initially filter down to using the filter model",
    )
    subparser.add_argument(
        "--retrieve_count",
        type=int,
        default=100,
        help="Number of snippets to retrieve after filtering",
    )
    subparser.add_argument(
        "--entire_file", action="store_true", help="Retrieve entire file contents"
    )
    subparser.add_argument(
        "--just_create_index",
        action="store_true",
        help="Create the index without performing retrieval",
    )

    subparser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Maximum number of concurrent workers",
    )


def build_prepare_data_parser(subparser: argparse.ArgumentParser):
    subparser.add_argument(
        "--dataset", type=str, default="princeton-nlp/SWE-bench_Multimodal"
    )
    subparser.add_argument("--split", type=str, default="dev")


def handle_get_steps(results_dir: Path, logs_dir: Path, args: argparse.Namespace):
    pass


def handle_retrieve_swe(results_dir: Path, logs_dir: Path, args: argparse.Namespace):
    retrieve_swe(results_dir, logs_dir, args)


def handle_prepare_data(results_dir: Path, logs_dir: Path, args: argparse.Namespace):
    prepare_data(results_dir, args)


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory for results and logs",
    )

    subparser = parser.add_subparsers(dest="command", required=True)

    get_steps_parser = subparser.add_parser("get_steps")
    build_get_steps_parser(get_steps_parser)
    get_steps_parser.set_defaults(func=handle_get_steps)

    retrieve_swe_parser = subparser.add_parser("retrieve_swe")
    build_retrieve_swe_parser(retrieve_swe_parser)
    retrieve_swe_parser.set_defaults(func=handle_retrieve_swe)

    prepare_data_parser = subparser.add_parser("prepare_data")
    build_prepare_data_parser(prepare_data_parser)
    prepare_data_parser.set_defaults(func=handle_prepare_data)

    return parser


def setup_results_dir(args: argparse.Namespace):
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def setup_logging(results_dir: Path):
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    root_log_file_name = "repstep.log"
    root_log_file_path = logs_dir / root_log_file_name

    logging.basicConfig(
        filename=root_log_file_path,
        format=LOGGING_FORMAT,
        level=ROOT_LOGGING_LEVEL,
    )

    return logs_dir


def main(argv=None):
    load_dotenv()

    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    results_dir = setup_results_dir(args)
    logs_dir = setup_logging(results_dir)
    args.func(results_dir, logs_dir, args)


if __name__ == "__main__":
    main()
