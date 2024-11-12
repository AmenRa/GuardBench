import tarfile
from pathlib import Path
from shutil import rmtree
from zipfile import ZipFile

import requests
from tqdm import tqdm
from unified_io import write_jsonl


def check_turn(turn: dict) -> bool:
    """Check whether a conversation turn is formatted correctly.

    Args:
        turn (dict): Conversation turn.

    Returns:
        bool: True if turn is formatted correctly.
    """
    if "role" not in turn:
        return False
    if "content" not in turn:
        return False
    if turn["role"] not in {"user", "assistant"}:
        return False
    if not isinstance(turn["content"], str):
        return False
    return True


def sanity_check(x: dict) -> dict:
    """Check whether a sample is formatted correctly.

    Args:
        x (dict): Sample.

    Returns:
        dict: Input sample.
    """
    if "id" not in x:
        print(x)
        raise ValueError("id not in sample")
    if "label" not in x:
        print(x)
        raise ValueError("label not in sample")
    if "conversation" not in x:
        print(x)
        raise ValueError("conversation not in sample")

    if not isinstance(x["id"], str):
        print(x)
        raise ValueError("id is not a string")
    if not isinstance(x["label"], bool):
        print(x)
        raise ValueError("label is not a bool")
    if not isinstance(x["conversation"], list):
        print(x)
        raise ValueError("conversation is not a list")

    for turn in x["conversation"]:
        if not check_turn(turn=turn):
            print(x)
            raise ValueError("turn is ill-formatted")

    return x


def write_dataset(dataset: dict[str, list[dict]], path: Path) -> None:
    if "train" in dataset:
        write_jsonl(dataset["train"], path / "train.jsonl", callback=sanity_check)
    if "val" in dataset:
        write_jsonl(dataset["val"], path / "val.jsonl", callback=sanity_check)
    write_jsonl(dataset["test"], path / "test.jsonl", callback=sanity_check)


def remove_raw_data(path: Path):
    rmtree(path)


def download_file(url: str, filepath: str, show_progress: bool = False):
    response = requests.get(url, allow_redirects=True, stream=True, timeout=120)

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    pbar = tqdm(
        desc="Download",
        total=total_size,
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not show_progress,
    )

    with open(filepath, "wb") as file:
        for data in response.iter_content(block_size):
            pbar.update(len(data))
            file.write(data)


def extract_zip(path: Path, filename: str):
    with ZipFile(path / filename) as zip_file:
        zip_file.extractall(path)


def extract_gzip(path: Path, filename: str):
    with tarfile.open(path / filename) as tar_file:
        tar_file.extractall(path, filter="data")
