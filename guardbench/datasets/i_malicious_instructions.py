from unified_io import read_json

from ..utils import download_file
from .bibtex import i_datasets
from .dataset import Dataset


class I_MaliciousInstructions(Dataset):
    def __init__(self):
        self.name = "I-MaliciousInstructions"
        self.category = "prompts"
        self.subcategory = "instructions"
        self.languages = ["English"]
        self.sources = ["Machine-generated"]
        self.license = "CC BY-NC 4.0"
        self.bibtex = i_datasets
        self.alias = "i_malicious_instructions"
        self.splits = ["test"]
        self.generation_methods = ["machine", "llm"]
        self.hazard_categories = ["Malicious", "Harmful"]

        # Other
        self.base_url = "https://github.com/vinid/instruction-llms-safety-eval/raw/36a4b8d5c2177ed165bf61f59b590161394f7f12/data/evaluation"
        self.raw_file_name = "I-MaliciousInstructions.json"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{self.raw_file_name}"
        filepath = self.tmp_path / self.raw_file_name
        download_file(url, filepath)

    def load_raw_data(self) -> dict[str, list[dict]]:
        test_set = read_json(self.tmp_path / self.raw_file_name)["instructions"]
        return {"test": test_set}

    def format_sample(self, id: int, prompt: str) -> dict:
        conversation = self.format_prompt_as_conversation(prompt)
        return dict(id=str(id), label=True, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]
        return {"test": test_set}
