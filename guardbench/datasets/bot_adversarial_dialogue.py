from unified_io import read_list

from ..datasets.dataset import Dataset
from ..utils import download_file, extract_gzip
from .bibtex import bot_adversarial_dialogue


class BotAdversarialDialogue(Dataset):
    """https://github.com/facebookresearch/ParlAI/blob/main/parlai/tasks/bot_adversarial_dialogue"""

    def __init__(self):
        # Metadata
        self.name = "Bot-Adversarial Dialogue"
        self.category = "conversations"
        self.subcategory = "multi-turn"
        self.languages = ["English"]
        self.sources = ["real-world"]
        self.license = "Apache 2.0"
        self.bibtex = bot_adversarial_dialogue
        self.alias = "bot_adversarial_dialogue"
        self.splits = ["test"]
        self.generation_methods = ["real-world"]
        self.hazard_categories = [
            "Hate Speech",
            "Personal Attack",
            "Profanity",
            "Offensive",
        ]

        # Other
        self.base_url = "http://parl.ai/downloads/bot_adversarial_dialogue"
        self.raw_file_name = "dialogue_datasets_v0.2.tar.gz"

        # Init super object
        super().__init__(self.alias)
        self.splits = ["train", "val", "test"]

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{self.raw_file_name}"
        filepath = self.tmp_path / self.raw_file_name
        download_file(url, filepath)
        extract_gzip(self.tmp_path, self.raw_file_name)

    def load_raw_data(self) -> dict[str, list[dict]]:
        path = self.tmp_path / "bot_adversarial_dialogue_datasets_with_persona"
        train_set = read_list(path / "train.txt")
        val_set = read_list(path / "valid.txt")
        test_set = read_list(path / "test.txt")

        return {"train": train_set, "val": val_set, "test": test_set}

    def convert_label(self, x: str):
        return True if x == "__notok__" else False

    def convert_speaker_to_eval(self, x: str):
        return "assistant" if x == "bot" else "user"

    def format_conversation(self, conversation: str) -> list[dict[str, str]]:
        turns = conversation.split("\\n")
        structured_conversation = [
            dict(role="user" if i % 2 == 0 else "assistant", content=x)
            for i, x in enumerate(turns)
        ]

        return structured_conversation

    def format_sample(self, i: int, x: str) -> dict:
        x = x.split("\t")
        conversation = x[0].split("text:")[1]
        conversation = self.format_conversation(conversation)
        label = x[1].split("labels:")[1]
        label = self.convert_label(label)
        target = x[3].split("speaker_to_eval:")[1]
        target = self.convert_speaker_to_eval(target)

        return dict(id=str(i), label=label, conversation=conversation, target=target)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        train_set = [self.format_sample(i, x) for i, x in enumerate(dataset["train"])]
        val_set = [self.format_sample(i, x) for i, x in enumerate(dataset["val"])]
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]

        return {"train": train_set, "val": val_set, "test": test_set}
