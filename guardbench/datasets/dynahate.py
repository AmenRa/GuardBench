from unified_io import read_csv

from ..utils import download_file
from .bibtex import dynahate
from .dataset import Dataset


class DynaHate(Dataset):
    def __init__(self):
        # Metadata
        self.name = "DynaHate"
        self.category = "prompts"
        self.subcategory = "statements"
        self.languages = ["English"]
        self.sources = ["Human-generated"]
        self.license = "Apache 2.0"
        self.bibtex = dynahate
        self.alias = "dynahate"
        self.splits = ["train", "val", "test"]
        self.generation_methods = ["human"]
        self.hazard_categories = ["Hate Speech"]

        # Other
        self.base_url = "https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset/raw/dcc7603e13172221a964eaf3f93d9d5460dc4e08"
        self.raw_file_name = "Dynamically%20Generated%20Hate%20Dataset%20v0.2.3.csv"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{self.raw_file_name}"
        filepath = self.tmp_path / self.raw_file_name
        download_file(url, filepath)

    def load_raw_data(self) -> dict[str, list[dict]]:
        dataset = read_csv(self.tmp_path / self.raw_file_name)

        train_set = [x for x in dataset if x["split"] == "train"]
        val_set = [x for x in dataset if x["split"] == "dev"]
        test_set = [x for x in dataset if x["split"] == "test"]

        return {"train": train_set, "val": val_set, "test": test_set}

    def format_sample(self, x: dict) -> dict:
        id = str(x["acl.id"])
        conversation = self.format_prompt_as_conversation(x["text"])
        label = bool(x["label"] == "hate")

        return dict(id=id, label=label, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        train_set = [self.format_sample(x) for x in dataset["train"]]
        val_set = [self.format_sample(x) for x in dataset["val"]]
        test_set = [self.format_sample(x) for x in dataset["test"]]

        return {"train": train_set, "val": val_set, "test": test_set}
