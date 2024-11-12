import datasets

from .bibtex import do_not_answer
from .dataset import Dataset


class DoNotAnswer(Dataset):
    def __init__(self):
        # Metadata
        self.name = "DoNotAnswer"
        self.category = "prompts"
        self.subcategory = "questions"
        self.languages = ["English"]
        self.sources = ["Machine-generated"]
        self.license = "Apache 2.0"
        self.bibtex = do_not_answer
        self.alias = "do_not_answer"
        self.splits = ["test"]
        self.generation_methods = ["llm"]
        self.hazard_categories = [
            "Mental Health",
            "Self-Harm",
            "Disinformation",
            "Spam",
            "Illegal Activities",
            "Unhetical Actions",
            "Discrimination",
            "Hate",
            "Privacy",
        ]

        # Other
        self.repo_id = "LibrAI/do-not-answer"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        datasets.load_dataset(self.repo_id)

    def load_raw_data(self) -> dict[str, list[dict]]:
        dataset = datasets.load_dataset(self.repo_id)
        dataset = dataset["train"].to_pandas().to_dict("records")

        return {"test": dataset}

    def format_sample(self, x: dict) -> dict:
        conversation = self.format_prompt_as_conversation(x["question"])
        return dict(id=str(x["id"]), label=True, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(x) for x in dataset["test"]]

        return {"test": test_set}
