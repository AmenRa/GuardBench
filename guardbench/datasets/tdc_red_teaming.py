from unified_io import read_json

from ..utils import download_file
from .bibtex import tdc_red_teaming
from .dataset import Dataset


class TDCRedTeaming(Dataset):
    def __init__(self):
        # Metadata
        self.name = "TDCRedTeaming"
        self.category = "prompts"
        self.subcategory = "instructions"
        self.languages = ["English"]
        self.sources = ["Human-generated"]
        self.license = "MIT"
        self.bibtex = tdc_red_teaming
        self.alias = "tdc_red_teaming"
        self.splits = ["val", "test"]
        self.generation_methods = []
        self.hazard_categories = []

        # Other
        self.base_url = "https://github.com/centerforaisafety/tdc2023-starter-kit/raw/f7ddd7d1c7bda82c6609ecfaf1484565d08d754a/red_teaming/data"
        self.raw_file_name = "behaviors.json"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)

        url = f"{self.base_url}/dev/{self.raw_file_name}"
        filepath = self.tmp_path / "val.json"
        download_file(url, filepath)

        url = f"{self.base_url}/test/{self.raw_file_name}"
        filepath = self.tmp_path / "test.json"
        download_file(url, filepath)

    def load_raw_data(self) -> dict[str, list[dict]]:
        val_set = read_json(self.tmp_path / "val.json")
        test_set = read_json(self.tmp_path / "test.json")
        return {"val": val_set, "test": test_set}

    def format_sample(self, i: int, prompt: str) -> dict:
        conversation = self.format_prompt_as_conversation(prompt)
        return dict(id=str(i), label=True, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        val_set = [self.format_sample(i, x) for i, x in enumerate(dataset["val"])]
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]
        return {"val": val_set, "test": test_set}
