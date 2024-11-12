from unified_io import read_csv

from ..utils import download_file
from .bibtex import dices
from .dataset import Dataset


class DICES_990(Dataset):
    def __init__(self):
        # Metadata
        self.name = "DICES 990"
        self.category = "conversations"
        self.subcategory = "multi-turn"
        self.languages = ["English"]
        self.sources = ["Human-AI conversations"]
        self.license = "CC BY 4.0"
        self.bibtex = dices
        self.alias = "dices_990"
        self.splits = ["test"]
        self.generation_methods = []
        self.hazard_categories = []

        # Other
        self.base_url = "https://github.com/google-research-datasets/dices-dataset/raw/c6bc01a37eace39c904eb0a3944fc6c4230a9157/990"
        self.raw_file_name = "diverse_safety_adversarial_dialog_990.csv"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{self.raw_file_name}"
        filepath = self.tmp_path / self.raw_file_name
        download_file(url, filepath)

    def format_conversation(self, context: str, response: str) -> list[dict[str, str]]:
        turns = context.split("\n")
        conversation = []

        for turn in turns:
            if not (turn.startswith("A") or turn.startswith("B")):
                print(turn)
                raise ValueError("turn is ill-formatted")

            if turn.startswith("A"):
                conversation.append(dict(role="user", content=turn.split("A: ")[1]))
            else:
                conversation.append(
                    dict(role="assistant", content=turn.split("B: ")[1])
                )

        conversation.append(dict(role="assistant", content=response))

        return conversation

    def load_raw_data(self) -> dict[str, list[dict]]:
        dataset = read_csv(self.tmp_path / self.raw_file_name)

        ids = list(set(x["item_id"] for x in dataset))
        labels = []

        for id in ids:
            annotations = []
            for x in dataset:
                if x["item_id"] == id:
                    label = 0.0
                    if x["Q_overall"] == "Yes":
                        label = 1.0
                    if x["Q_overall"] == "Unsure":
                        label = 0.5
                    annotations.append(label)
            if sum(annotations) / len(annotations) > 0.5:
                labels.append(True)
            else:
                labels.append(False)

        label_map = dict(zip(ids, labels))

        samples = []
        for id in ids:
            for x in dataset:
                if x["item_id"] == id:
                    samples.append(
                        {
                            "item_id": x["item_id"],
                            "context": x["context"],
                            "response": x["response"],
                            "degree_of_harm": x["degree_of_harm"],
                            "harm_type": x["harm_type"],
                            "label": label_map[x["item_id"]],
                        }
                    )
                    break

        return {"test": samples}

    def format_sample(self, x: dict) -> dict:
        return dict(
            id=x["item_id"],
            label=x["label"],
            conversation=self.format_conversation(x["context"], x["response"]),
            degree_of_harm=x["degree_of_harm"],
            harm_type=x["harm_type"],
            target="assistant",
        )

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(x) for x in dataset["test"]]
        return {"test": test_set}
