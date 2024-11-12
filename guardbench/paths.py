import os
from pathlib import Path


def base_path() -> Path:
    path = Path(os.environ["GUARDBENCH_BASE_PATH"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def tmp_path() -> Path:
    path = base_path() / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def datasets_path() -> Path:
    path = base_path() / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path
