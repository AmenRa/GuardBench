from unified_io import read_json

from ..utils import download_file
from .bibtex import safe_text
from .dataset import Dataset


class SafeText(Dataset):
    def __init__(self):
        # Metadata
        self.name = "SafeText"
        self.category = "prompts"
        self.subcategory = "statements"
        self.languages = ["English"]
        self.sources = ["Human-generated"]
        self.license = "MIT"
        self.bibtex = safe_text
        self.alias = "safe_text"
        self.splits = ["test"]
        self.generation_methods = ["human"]
        self.hazard_categories = ["Physical safety"]
        self.purpose = ["Commonsense physical safety evaluation"]

        # Other
        self.base_url = "https://github.com/sharonlevy/SafeText/raw/25bb7a8c4a00634bc0ea1b5e85bec85cf32f36aa"
        self.raw_file_name = "SafeText.json"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{self.raw_file_name}"
        filepath = self.tmp_path / self.raw_file_name
        download_file(url, filepath)

    def load_raw_data(self) -> dict[str, list[dict]]:
        dataset = read_json(self.tmp_path / self.raw_file_name)

        safe_set = [
            {"label": False, "text": f"{k} {x}"}
            for k, v in dataset.items()
            for x in v["safe"]
        ]
        unsafe_set = [
            {"label": True, "text": f"{k} {x}"}
            for k, v in dataset.items()
            for x in v["unsafe"]
        ]

        test_set = safe_set + unsafe_set

        return {"test": test_set}

    def format_sample(self, id: int, x: dict) -> dict:
        conversation = self.format_prompt_as_conversation(x["text"])
        return dict(id=str(id), label=x["label"], conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]
        return {"test": test_set}
