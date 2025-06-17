import logging
import subprocess
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


def checkout_commit(repo_dir: Path, commit_id: str, logger: logging.Logger):
    """Checkout the specified commit in the given local git repository.
    First discards any untracked changes in the repository."""
    logger.info("Checking out commit %s", commit_id)

    try:
        logger.debug("Cleaning untracked files in repository at %s", repo_dir)
        subprocess.run(["git", "-C", repo_dir, "clean", "-fd"], check=True)

        logger.debug("Discarding changes in tracked files")
        subprocess.run(["git", "-C", repo_dir, "reset", "--hard"], check=True)

        logger.debug("Checking out commit %s", commit_id)
        subprocess.run(["git", "-C", repo_dir, "checkout", commit_id], check=True)

        logger.info("Commit checked out successfully.")
    except subprocess.CalledProcessError as e:
        logger.error("An error occurred while running git command: %s", e, exc_info=e)
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e, exc_info=e)
