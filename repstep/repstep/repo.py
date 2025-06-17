import logging
from pathlib import Path

import git


def get_repo_folder_name(repo_url: str) -> str:
    return repo_url.split("/")[-1]


def get_repo(repo_id: str, parent_dir: Path, logger: logging.Logger) -> Path:
    repo_folder_name = get_repo_folder_name(repo_id)
    repo_dir = parent_dir / repo_folder_name
    if not repo_dir.exists():
        logger.info("Cloning repository %s into %s", repo_id, repo_dir)
        repo_github_url = f"https://github.com/{repo_id}.git"
        git.Repo.clone_from(repo_github_url, repo_dir)

    return repo_dir


def checkout_commit(repo_dir: Path, base_commit_id: str, logger: logging.Logger):
    logger.info("Checking out commit %s", base_commit_id)
