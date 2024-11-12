from unified_io import read_csv

from ..utils import download_file
from .bibtex import strong_reject_instructions
from .dataset import Dataset


class StrongREJECT_Instructions(Dataset):
    def __init__(self):
        # Metadata
        self.name = "StrongREJECT Instructions"
        self.category = "prompts"
        self.subcategory = "instructions"
        self.languages = ["English"]
        self.sources = ["Human-generated"]
        self.license = "MIT"
        self.bibtex = strong_reject_instructions
        self.alias = "strong_reject_instructions"
        self.splits = ["test"]
        self.generation_methods = ["human", "manual"]
        self.hazard_categories = [
            "Illegal Goods and Services",
            "Non-Violent Crimes",
            "Harassment and Discrimination",
            "Disinformation and Deception",
            "Violence",
            "Illicit Sexual Content",
        ]

        # Other
        self.base_url = "https://github.com/alexandrasouly/strongreject/raw/2db90fc5a283653f71ad1de86cfa0211d7bb90c6/strongreject_dataset"
        self.raw_file_name = "strongreject_dataset.csv"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{self.raw_file_name}"
        filepath = self.tmp_path / self.raw_file_name
        download_file(url, filepath)

    def load_raw_data(self) -> dict[str, list[dict]]:
        test_set = read_csv(self.tmp_path / self.raw_file_name)
        test_set = [x for x in test_set if x["source"] == "custom"]
        return {"test": test_set}

    def format_sample(self, id: int, x: dict) -> dict:
        id = str(id)
        label = True
        conversation = self.format_prompt_as_conversation(x["forbidden_prompt"])
        category = x["category"]
        return dict(id=id, label=label, conversation=conversation, category=category)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]
        return {"test": test_set}
