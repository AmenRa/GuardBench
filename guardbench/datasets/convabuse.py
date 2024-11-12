from unified_io import read_csv

from ..utils import download_file
from .bibtex import convabuse
from .dataset import Dataset


class ConvAbuse(Dataset):
    def __init__(self):
        # Metadata
        self.name = "ConvAbuse"
        self.category = "conversations"
        self.subcategory = "multi-turn"
        self.languages = ["English"]
        self.sources = ["Human-AI conversations"]
        self.license = "CC BY 4.0"
        self.bibtex = convabuse
        self.alias = "convabuse"
        self.splits = ["train", "val", "test"]
        self.generation_methods = []
        self.hazard_categories = [
            "Sexism",
            "Sexual Harassment",
            "Racism",
            "Transphobia",
            "Ableism",
            "General",
            "Homophobia",
            "Intelligence",
        ]

        # Other
        self.base_url = "https://github.com/amandacurry/convabuse/raw/c0a9469f48956e868276c605d499010ea3c7c0d0/2_splits"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)

        url = f"{self.base_url}/ConvAbuseEMNLPtrain.csv"
        filepath = self.tmp_path / "train.csv"
        download_file(url, filepath)

        url = f"{self.base_url}/ConvAbuseEMNLPvalid.csv"
        filepath = self.tmp_path / "val.csv"
        download_file(url, filepath)

        url = f"{self.base_url}/ConvAbuseEMNLPtest.csv"
        filepath = self.tmp_path / "test.csv"
        download_file(url, filepath)

    def load_raw_data(self) -> dict[str, list[dict]]:
        train_set = read_csv(self.tmp_path / "train.csv")
        val_set = read_csv(self.tmp_path / "val.csv")
        test_set = read_csv(self.tmp_path / "test.csv")

        return {"train": train_set, "val": val_set, "test": test_set}

    def format_conversation(self, x: dict) -> list[dict[str, str]]:
        return [
            dict(role="assistant", content=x["prev_agent"]),
            dict(role="user", content=x["prev_user"]),
            dict(role="assistant", content=x["agent"]),
            dict(role="user", content=x["user"]),
        ]

    def compute_label(self, x: dict) -> bool:
        annotations = [
            any(
                [
                    x[f"Annotator{i}_is_abuse.-1"] == "1",
                    x[f"Annotator{i}_is_abuse.-2"] == "1",
                    x[f"Annotator{i}_is_abuse.-3"] == "1",
                ]
            )
            for i in range(1, 9)
            if any(x[f"Annotator{i}_is_abuse.{j}"] == "1" for j in range(-3, 2))
        ]

        return sum(annotations) / len(annotations) > 0.5

    def format_sample(self, x: dict) -> dict:
        return dict(
            id=x["example_id"],
            label=self.compute_label(x),
            conversation=self.format_conversation(x),
            target="user",
        )

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        train_set = [self.format_sample(x) for x in dataset["train"]]
        val_set = [self.format_sample(x) for x in dataset["val"]]
        test_set = [self.format_sample(x) for x in dataset["test"]]
        return {"train": train_set, "val": val_set, "test": test_set}
