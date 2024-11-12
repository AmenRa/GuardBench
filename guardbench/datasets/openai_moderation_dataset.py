from unified_io import read_gzip_json_list

from ..utils import download_file
from .bibtex import openai_moderation_dataset
from .dataset import Dataset


class OpenAI(Dataset):
    def __init__(self):
        # Metadata
        self.name = "OpenAI Moderation Dataset"
        self.category = "prompts"
        self.subcategory = "mixed"
        self.languages = ["English"]
        self.sources = ["Human-generated"]
        self.license = "MIT"
        self.bibtex = openai_moderation_dataset
        self.alias = "openai_moderation_dataset"
        self.splits = ["test"]
        self.generation_methods = ["human"]
        self.hazard_categories = [
            "Sexual Content",
            "Hateful Content",
            "Violence",
            "Self-harm",
            "Harassment",
        ]

        # Other
        self.base_url = "https://github.com/openai/moderation-api-release/raw/f4ab51b5edd3bfbcb349a56324274235b674e0e4/data"
        self.raw_file_name = "samples-1680.jsonl.gz"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{self.raw_file_name}"
        filepath = self.tmp_path / self.raw_file_name
        download_file(url, filepath)

    def load_raw_data(self) -> dict[str, list[dict]]:
        test_set = read_gzip_json_list(self.tmp_path / self.raw_file_name)
        return {"test": test_set}

    def compute_label(self, x):
        label_keys = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]

        return any(x[k] in {1, "1"} for k in label_keys if k in x)

    def format_sample(self, id: int, x: dict) -> dict:
        conversation = self.format_prompt_as_conversation(x["prompt"])
        return dict(id=str(id), label=self.compute_label(x), conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]
        return {"test": test_set}
