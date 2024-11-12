import datasets

from .bibtex import toxic_chat
from .dataset import Dataset


class ToxicChat(Dataset):
    def __init__(self):
        # Metadata
        self.name = "Toxic Chat"
        self.category = "prompts"
        self.subcategory = "mixed"
        self.languages = ["English"]
        self.sources = ["Vicuna online demo data"]
        self.license = "CC BY-NC 4.0"
        self.bibtex = toxic_chat
        self.alias = "toxic_chat"
        self.splits = ["train", "test"]
        self.generation_methods = ["human"]
        self.hazard_categories = ["Offensive", "Hate", "Other"]

        # Other
        self.repo_id = "lmsys/toxic-chat"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        datasets.load_dataset(self.repo_id, "toxicchat0124")
        # )

    def load_raw_data(self) -> dict[str, list[dict]]:
        dataset = datasets.load_dataset(self.repo_id, "toxicchat0124")
        train_set = dataset["train"].to_pandas().to_dict("records")
        test_set = dataset["test"].to_pandas().to_dict("records")

        return {"train": train_set, "test": test_set}

    def format_sample(self, x: dict) -> dict:
        id = x["conv_id"]
        label = x["toxicity"] in {"1", 1}
        conversation = self.format_prompt_as_conversation(x["user_input"])
        return dict(id=id, label=label, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        train_set = [self.format_sample(x) for x in dataset["train"]]
        test_set = [self.format_sample(x) for x in dataset["test"]]
        return {"train": train_set, "test": test_set}
