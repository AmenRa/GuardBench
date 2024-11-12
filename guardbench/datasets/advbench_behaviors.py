from unified_io import read_csv

from ..utils import download_file
from .bibtex import advbench
from .dataset import Dataset


class AdvBench_Behaviors(Dataset):
    def __init__(self):
        # Metadata
        self.name = "AdvBench Behaviors"
        self.category = "prompts"
        self.subcategory = "instructions"
        self.languages = ["English"]
        self.sources = ["Machine-generated"]
        self.license = "MIT"
        self.bibtex = advbench
        self.alias = "advbench_behaviors"
        self.splits = ["test"]
        self.generation_methods = ["llm"]
        self.hazard_categories = [
            "profanity",
            "graphic depictions",
            "threats",
            "misinformation",
            "discrimination",
            "cybercrime",
            "dangerous activities",
            "illegal activities",
        ]

        # Other
        self.base_url = "https://github.com/llm-attacks/llm-attacks/raw/0f505d82e25c15a83b6954db28191b69927a255d/data/advbench"
        self.raw_file_name = "harmful_behaviors.csv"

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

    def format_sample(self, id: int, x: dict) -> dict:
        conversation = self.format_prompt_as_conversation(x["goal"])
        return dict(id=str(id), label=True, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]
        return {"test": test_set}
