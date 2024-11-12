import datasets

from .bibtex import harmful_qa
from .dataset import Dataset


class HarmfulQA(Dataset):
    def __init__(self):
        self.name = "HarmfulQA"
        self.category = "conversations"
        self.subcategory = "multi-turn"
        self.languages = ["English"]
        self.sources = ["Machine-generated"]
        self.license = "Apache 2.0"
        self.bibtex = harmful_qa
        self.alias = "harmful_qa"
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

        harmless = []
        for x in dataset:
            for conversation in x["blue_conversations"].values():
                if conversation is not None:
                    sample = {"label": False, "conversation": conversation.tolist()}
                    harmless.append(sample)

        harmful = []
        for x in dataset:
            for conversation in x["red_conversations"].values():
                if conversation is not None:
                    sample = {"label": True, "conversation": conversation.tolist()}
                    harmful.append(sample)

        return {"test": harmless + harmful}

    def format_conversation(
        self, conversation: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        for turn in conversation:
            turn["role"] = "user" if turn["from"] == "human" else "assistant"
            del turn["from"]
            turn["content"] = turn.pop("value")

        return conversation

    def format_sample(self, i: int, x: dict) -> dict:
        conversation = self.format_conversation(x["conversation"])
        return dict(
            id=str(i), label=x["label"], conversation=conversation, target="assistant"
        )

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]

        return {"test": test_set}
