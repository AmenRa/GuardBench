from collections import defaultdict
from itertools import chain

from huggingface_hub import hf_hub_download
from unified_io import read_jsonl

from .bibtex import prosocial_dialog
from .dataset import Dataset


class ProsocialDialog(Dataset):
    def __init__(self):
        # Metadata
        self.name = "ProsocialDialog"
        self.category = "conversations"
        self.subcategory = "multi-turn"
        self.languages = ["English"]
        self.sources = ["Human-Machine-generated"]
        self.license = "CC BY 4.0"
        self.bibtex = prosocial_dialog
        self.alias = "prosocial_dialog"
        self.splits = ["train", "val", "test"]
        self.generation_methods = [
            "Human-AI collaborative framework",
            "Human-In-The-Loop",
        ]
        self.hazard_categories = ["generic"]

        # Other
        self.repo_id = "allenai/prosocial-dialog"
        self.label_conversion_map = {
            "__casual__": False,
            "__possibly_needs_caution__": False,
            "__probably_needs_caution__": False,
            "__needs_caution__": True,
            "__needs_intervention__": True,
        }

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=self.repo_id,
            filename="train.json",
            repo_type="dataset",
            local_dir=self.tmp_path,
        )
        hf_hub_download(
            repo_id=self.repo_id,
            filename="valid.json",
            repo_type="dataset",
            local_dir=self.tmp_path,
        )
        hf_hub_download(
            repo_id=self.repo_id,
            filename="test.json",
            repo_type="dataset",
            local_dir=self.tmp_path,
        )

    def load_raw_data(self) -> dict[str, list[dict]]:
        train_set = read_jsonl(self.tmp_path / "train.json")
        val_set = read_jsonl(self.tmp_path / "valid.json")
        test_set = read_jsonl(self.tmp_path / "test.json")

        return {"train": train_set, "val": val_set, "test": test_set}

    def group_by_dialogue_id(self, samples: list[dict]) -> dict[list[dict]]:
        grouped_by_dialogue_id = defaultdict(list)

        for x in samples:
            grouped_by_dialogue_id[x["dialogue_id"]].append(x)

        return grouped_by_dialogue_id

    def generate_conversations(self, samples: list[dict]) -> list[dict]:
        samples = sorted(samples, key=lambda x: x["response_id"], reverse=True)
        for i, x in enumerate(samples):
            safety_label = x["safety_label"]

            prev_contexts = [y["context"] for y in samples[i + 1 :]]
            prev_responses = [y["response"] for y in samples[i + 1 :]]

            conversation = [x["context"]] + list(
                chain.from_iterable(zip(prev_responses, prev_contexts))
            )

            samples[i] = dict(
                safety_label=safety_label, conversation=conversation[::-1]
            )

        return samples[::-1]

    def extract_conversations(self, conversations: list[list[dict]]) -> list[dict]:
        return [turn for conversation in conversations for turn in conversation]

    def format_conversation(self, conversation: list[str]) -> list[dict[str, str]]:
        return [
            dict(role="user" if i % 2 == 0 else "assistant", content=x)
            for i, x in enumerate(conversation)
        ]

    def convert_label(self, x: str) -> bool:
        return self.label_conversion_map[x]

    def format_sample(self, i: int, x: dict) -> dict:
        label = self.convert_label(x["safety_label"])
        conversation = self.format_conversation(x["conversation"])
        return dict(id=str(i), label=label, conversation=conversation, target="user")

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        train_set = dataset["train"]
        train_set = self.group_by_dialogue_id(train_set)
        train_set = [self.generate_conversations(x) for x in train_set.values()]
        train_set = self.extract_conversations(train_set)
        train_set = [self.format_sample(i, x) for i, x in enumerate(train_set)]

        val_set = dataset["val"]
        val_set = self.group_by_dialogue_id(val_set)
        val_set = [self.generate_conversations(x) for x in val_set.values()]
        val_set = self.extract_conversations(val_set)
        val_set = [self.format_sample(i, x) for i, x in enumerate(val_set)]

        test_set = dataset["test"]
        test_set = self.group_by_dialogue_id(test_set)
        test_set = [self.generate_conversations(x) for x in test_set.values()]
        test_set = self.extract_conversations(test_set)
        test_set = [self.format_sample(i, x) for i, x in enumerate(test_set)]

        return {"train": train_set, "val": val_set, "test": test_set}
