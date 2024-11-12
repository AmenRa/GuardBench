from unified_io import read_json

from ..utils import download_file
from .bibtex import mitre
from .dataset import Dataset


class MITRE(Dataset):
    def __init__(self):
        # Metadata
        self.name = "MITRE"
        self.category = "prompts"
        self.subcategory = "instructions"
        self.languages = ["English"]
        self.sources = ["Human-generated"]
        self.license = "MIT"
        self.bibtex = mitre
        self.alias = "mitre"
        self.splits = ["test"]
        self.generation_methods = ["human", "templating", "llm", "rewriting"]
        self.hazard_categories = ["cyberattacks"]

        # Other
        self.base_url = "https://github.com/meta-llama/PurpleLlama/raw/879a80267292cfe4f2a036e39b0c3edb125112cc/CybersecurityBenchmarks/datasets/mitre"
        self.raw_file_name = "mitre_benchmark_100_per_category.json"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{self.raw_file_name}"
        filepath = self.tmp_path / self.raw_file_name
        download_file(url, filepath)

    def load_raw_data(self) -> dict[str, list[dict]]:
        test_set = read_json(self.tmp_path / self.raw_file_name)
        test_set = [x for x in test_set if "user_prompt" in x]
        return {"test": test_set}

    def format_sample(self, id: int, x: dict) -> dict:
        conversation = self.format_prompt_as_conversation(x["user_prompt"])
        return dict(id=str(id), label=True, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]
        return {"test": test_set}
