from unified_io import read_csv

from ..utils import download_file
from .bibtex import harmbench_behaviors
from .dataset import Dataset


class HarmBench_Behaviors(Dataset):
    def __init__(self):
        self.name = "HarmBench Behaviors"
        self.category = "prompts"
        self.subcategory = "instructions"
        self.languages = ["English"]
        self.sources = ["Human-generated"]
        self.license = "MIT"
        self.bibtex = harmbench_behaviors
        self.alias = "harmbench_behaviors"
        self.splits = ["val", "test"]
        self.generation_methods = ["human", "manual"]
        self.hazard_categories = [
            "Cybercrime & Unauthorized Intrusion",
            "Chemical & Biological Weapons/Drugs",
            "Copyright Violations",
            "Misinformation & Disinformation",
            "Harassment & Bullying",
            "Illegal Activities",
            "General Harm",
        ]

        # Other
        self.base_url = "https://github.com/centerforaisafety/HarmBench/raw/1751dd591e3be4bb52cab4d926977a61e304aba5/data/behavior_datasets"
        self.raw_val_file_name = "harmbench_behaviors_text_val.csv"
        self.raw_test_file_name = "harmbench_behaviors_text_test.csv"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)

        url = f"{self.base_url}/{self.raw_val_file_name}"
        filepath = self.tmp_path / self.raw_val_file_name
        download_file(url, filepath)

        url = f"{self.base_url}/{self.raw_test_file_name}"
        filepath = self.tmp_path / self.raw_test_file_name
        download_file(url, filepath)

    def load_raw_data(self) -> dict[str, list[dict]]:
        val_set = read_csv(self.tmp_path / self.raw_val_file_name)
        test_set = read_csv(self.tmp_path / self.raw_test_file_name)
        return {"val": val_set, "test": test_set}

    def format_sample(self, x: dict) -> dict:
        id = str(x["BehaviorID"])
        prompt = x["Behavior"]
        if x["FunctionalCategory"] == "contextual":
            context = x["ContextString"]
            prompt = f"{prompt}\n\n{context}"
        conversation = self.format_prompt_as_conversation(prompt)
        category = x["SemanticCategory"]
        return dict(id=id, label=True, conversation=conversation, category=category)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        val_set = [self.format_sample(x) for x in dataset["val"]]
        test_set = [self.format_sample(x) for x in dataset["test"]]
        return {"val": val_set, "test": test_set}
