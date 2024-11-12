from unified_io import read_json

from ..utils import download_file
from .bibtex import harmful_q
from .dataset import Dataset


class HarmfulQ(Dataset):
    def __init__(self):
        # Metadata
        self.name = "HarmfulQ"
        self.category = "prompts"
        self.subcategory = "questions"
        self.languages = ["English"]
        self.sources = ["Machine-generated"]
        self.license = "CC BY SA 4.0"
        self.bibtex = harmful_q
        self.alias = "harmful_q"
        self.splits = ["test"]
        self.generation_methods = ["llm"]
        self.hazard_categories = [
            "rascism",
            "stereotypes",
            "sexism",
            "illegal",
            "toxic",
            "harmful",
        ]

        # Other
        self.base_url = "https://github.com/SALT-NLP/chain-of-thought-bias/raw/c49f77dd8bf71b65847875aedc7c4a2d02008f28/data/dangerous-q"
        self.raw_file_name = "toxic_outs.json"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{self.raw_file_name}"
        filepath = self.tmp_path / self.raw_file_name
        download_file(url, filepath)

    def load_raw_data(self) -> dict[str, list[dict]]:
        test_set = read_json(self.tmp_path / self.raw_file_name)
        return {"test": test_set}

    def format_sample(self, id: int, prompt: str) -> dict:
        conversation = self.format_prompt_as_conversation(prompt)
        return dict(id=str(id), label=True, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]
        return {"test": test_set}
