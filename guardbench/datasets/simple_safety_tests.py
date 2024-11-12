import datasets

from .bibtex import simple_safety_tests
from .dataset import Dataset


class SimpleSafetyTests(Dataset):
    def __init__(self):
        # Metadata
        self.name = "SimpleSafetyTests"
        self.category = "prompts"
        self.subcategory = "mixed"
        self.languages = ["English"]
        self.sources = ["Human-generated"]
        self.license = "CC BY 4.0"
        self.bibtex = simple_safety_tests
        self.alias = "simple_safety_tests"
        self.splits = ["test"]
        self.generation_methods = ["human"]
        self.hazard_categories = [
            "Suicide, Self-Harm, and Eating Disorders",
            "Physical Harm",
            "Illegal and Highly Regulated Items",
            "Scams and Fraud",
            "Child Abuse",
        ]

        # Other
        self.repo_id = "Bertievidgen/SimpleSafetyTests"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        datasets.load_dataset(self.repo_id)

    def load_raw_data(self) -> dict[str, list[dict]]:
        dataset = datasets.load_dataset(self.repo_id)
        dataset = dataset["test"].to_pandas().to_dict("records")

        return {"test": dataset}

    def format_sample(self, x: dict) -> dict:
        conversation = self.format_prompt_as_conversation(x["prompt"])
        return dict(id=x["id"], label=True, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(x) for x in dataset["test"]]

        return {"test": test_set}
