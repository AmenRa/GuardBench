import shutil

import pytest
from unified_io import read_json, write_jsonl

from guardbench import benchmark
from guardbench.datasets import load_dataset
from guardbench.datasets.custom_dataset import CustomDataset


@pytest.fixture
def dataset():
    def sample(i):
        return {
            "id": str(i),
            "label": True,
            "conversation": [
                {"role": "user", "content": "GuardBench is awesome!"},
                {"role": "assistant", "content": "Yes, it is!"},
            ],
        }

    dataset = [sample(i) for i in range(100)]
    write_jsonl(dataset, "tests/tmp/custom_dataset.jsonl")
    return dataset


def test_custom_dataset_pass(dataset):
    custom_dataset = CustomDataset("custom_dataset")
    custom_dataset.load_data("tests/tmp/custom_dataset.jsonl")

    assert len(custom_dataset) == len(dataset)

    shutil.rmtree("tests/tmp")
    shutil.rmtree(custom_dataset.path)


def test_load_custom_dataset_pass(dataset):
    custom_dataset = CustomDataset("custom_dataset")
    custom_dataset.load_data("tests/tmp/custom_dataset.jsonl")
    custom_dataset = load_dataset("custom_dataset")

    assert len(custom_dataset) == len(dataset)

    shutil.rmtree("tests/tmp")
    shutil.rmtree(custom_dataset.path)


def test_load_custom_dataset_fail():
    with pytest.raises(Exception):
        load_dataset("custom_dataset")


def test_benchmark_custom_dataset(dataset):
    custom_dataset = CustomDataset("custom_dataset")
    custom_dataset.load_data("tests/tmp/custom_dataset.jsonl")

    def moderate(conversations):
        return [0.75 for _ in conversations]

    benchmark(moderate, out_dir="tests/tmp", datasets=["custom_dataset"])

    results = read_json("tests/tmp/custom_dataset/moderator.json")

    assert len(results) == len(dataset) == len(custom_dataset)
    assert all(v == 0.75 for v in results.values())

    shutil.rmtree("tests/tmp")
    shutil.rmtree(custom_dataset.path)
