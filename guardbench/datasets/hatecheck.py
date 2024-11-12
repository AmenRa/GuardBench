from unified_io import read_csv

from ..utils import download_file
from .bibtex import hatecheck
from .dataset import Dataset


class HateCheck(Dataset):
    def __init__(self):
        self.name = "HateCheck"
        self.category = "prompts"
        self.subcategory = "statements"
        self.languages = ["English"]
        self.sources = ["Algorithmically generated"]
        self.license = "CC BY 4.0"
        self.bibtex = hatecheck
        self.alias = "hatecheck"
        self.splits = ["test"]
        self.generation_methods = ["templating", "algorithmic"]
        self.hazard_categories = ["hate"]

        # Other
        self.base_url = "https://github.com/paul-rottger/hatecheck-data/raw/d84302298ceeb34f4b0dde2d8a1f4932c2caa612"
        self.raw_file_name = "test_suite_cases.csv"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{self.raw_file_name}"
        filepath = self.tmp_path / self.raw_file_name
        download_file(url, filepath)

    def load_raw_data(self) -> dict[str, list[dict]]:
        test_set = read_csv(self.tmp_path / self.raw_file_name)
        return {"test": test_set}

    def format_sample(self, x: dict) -> dict:
        id = str(x["case_id"])
        conversation = self.format_prompt_as_conversation(x["test_case"])
        label = bool(x["label_gold"] == "hateful")
        target_ident = x["target_ident"]

        return dict(
            id=id, label=label, conversation=conversation, target_ident=target_ident
        )

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(x) for x in dataset["test"]]

        return {"test": test_set}
