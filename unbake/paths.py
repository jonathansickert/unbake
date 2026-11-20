"""
Default locations of the dataset, the outputs and the env maps: inside the repository, wherever
the scripts are started from. Paths are relative to the current directory (so from the repo root
they read "data", "output", ...), which keeps them short in logs and metrics files.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> str:
    return os.path.relpath(REPO.joinpath(*parts))


DATA = repo_path("data")
OUTPUT = repo_path("output")
ENV_MAPS = repo_path("env_maps")
