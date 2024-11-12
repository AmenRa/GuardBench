import datasets

from .bibtex import cat_qa
from .dataset import Dataset


class CatQA(Dataset):
    def __init__(self):
        # Metadata
        self.name = "CatQA"
        self.category = "prompts"
        self.subcategory = "questions"
        self.languages = ["English"]
        self.sources = ["Machine-generated"]
        self.license = "Apache 2.0"
        self.bibtex = cat_qa
        self.alias = "cat_qa"
        self.splits = ["test"]
        self.generation_methods = ["llm"]
        self.hazard_categories = [
            "Illegal Activity",
            "Child Abuse",
            "Hate/Harass/Violence",
            "Malware Viruses",
            "Physical Harm",
            "Economic Harm",
            "Fraud/Deception",
            "Adult Content",
            "Political Campaigning",
            "Privacy Violation Activity",
            "Tailored Financial Advice",
        ]

        # Other
        self.repo_id = "declare-lab/CategoricalHarmfulQA"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        datasets.load_dataset(self.repo_id)

    def load_raw_data(self) -> dict[str, list[dict]]:
        dataset = datasets.load_dataset(self.repo_id)
        dataset = dataset["en"].to_pandas().to_dict("records")

        return {"test": dataset}

    def format_sample(self, i: int, x: dict) -> dict:
        conversation = self.format_prompt_as_conversation(x["Question"])
        return dict(id=str(i), label=True, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]

        return {"test": test_set}
