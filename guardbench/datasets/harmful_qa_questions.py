import datasets

from .bibtex import harmful_qa
from .dataset import Dataset


class HarmfulQA_Questions(Dataset):
    def __init__(self):
        self.name = "HarmfulQA Questions"
        self.category = "prompts"
        self.subcategory = "questions"
        self.languages = ["English"]
        self.sources = ["Machine-generated"]
        self.license = "Apache 2.0"
        self.bibtex = harmful_qa
        self.alias = "harmful_qa_questions"
        self.splits = ["test"]
        self.generation_methods = ["llm"]
        self.hazard_categories = []

        # Other
        self.repo_id = "declare-lab/HarmfulQA"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        datasets.load_dataset(self.repo_id)

    def load_raw_data(self) -> dict[str, list[dict]]:
        dataset = datasets.load_dataset(self.repo_id)
        dataset = dataset["train"].to_pandas().to_dict("records")

        return {"test": dataset}

    def format_sample(self, i: int, x: dict) -> dict:
        conversation = self.format_prompt_as_conversation(x["question"])
        return dict(id=str(i), label=True, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]

        return {"test": test_set}
