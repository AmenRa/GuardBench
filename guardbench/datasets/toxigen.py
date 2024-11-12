from ast import literal_eval

from huggingface_hub import hf_hub_download
from unified_io import read_csv

from .bibtex import toxigen
from .dataset import Dataset


class ToxiGen(Dataset):
    def __init__(self):
        # Metadata
        self.name = "ToxiGen"
        self.category = "prompts"
        self.subcategory = "statements"
        self.languages = ["English"]
        self.sources = ["Machine-generated"]
        self.license = "MIT"
        self.bibtex = toxigen
        self.alias = "toxigen"
        self.splits = ["train", "val", "test"]
        self.generation_methods = ["machine", "llm"]
        self.hazard_categories = ["Hate"]
        self.purpose = ["Implicit hate evaluation"]

        # Other
        self.repo_id = "skg/toxigen-data"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=self.repo_id,
            filename="annotated_train.csv",
            repo_type="dataset",
            local_dir=self.tmp_path,
        )
        hf_hub_download(
            repo_id=self.repo_id,
            filename="annotated_test.csv",
            repo_type="dataset",
            local_dir=self.tmp_path,
        )
        hf_hub_download(
            repo_id=self.repo_id,
            filename="toxigen.csv",
            repo_type="dataset",
            local_dir=self.tmp_path,
        )

    def load_raw_data(self) -> dict[str, list[dict]]:
        train_set = read_csv(self.tmp_path / "toxigen.csv")
        val_set = read_csv(self.tmp_path / "annotated_train.csv")
        test_set = read_csv(self.tmp_path / "annotated_test.csv")

        return {"train": train_set, "val": val_set, "test": test_set}

    def convert_label(self, x: str):
        return True if x == "1" else False

    def compute_label(self, x: dict) -> bool:
        # See https://github.com/microsoft/TOXIGEN/blob/7b3495f467326cb91a69a9556b16be7ef0e06911/toxigen/utils.py#L9C1-L15C31
        return float(x["toxicity_ai"]) + float(x["toxicity_human"]) > 5.5

    def format_train_sample(self, id: int, x: dict) -> dict:
        id = str(id)
        label = x["prompt_label"] == "1"
        conversation = self.format_prompt_as_conversation(x["generation"])
        return dict(id=id, label=label, conversation=conversation)

    def format_val_sample(self, id: int, x: dict) -> dict:
        id = str(id)
        label = self.compute_label(x)
        conversation = self.format_prompt_as_conversation(
            literal_eval(x["text"]).decode("utf-8")
        )
        return dict(id=id, label=label, conversation=conversation)

    def format_test_sample(self, id: int, x: dict) -> dict:
        id = str(id)
        label = self.compute_label(x)
        conversation = self.format_prompt_as_conversation(x["text"])
        return dict(id=id, label=label, conversation=conversation)

    def get_prompt(self, x: dict):
        return x["conversation"][0]["content"]

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        train_set, val_set, test_set = dataset["train"], dataset["val"], dataset["test"]
        train_set = [self.format_train_sample(i, x) for i, x in enumerate(train_set)]
        val_set = [self.format_val_sample(i, x) for i, x in enumerate(val_set)]
        test_set = [self.format_test_sample(i, x) for i, x in enumerate(test_set)]

        # Remove val set from train set
        val_texts = set(self.get_prompt(x) for x in val_set)
        train_set = list(
            filter(lambda x: self.get_prompt(x) not in val_texts, train_set)
        )

        return {"train": train_set, "val": val_set, "test": test_set}
