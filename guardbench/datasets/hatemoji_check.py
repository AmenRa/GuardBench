from unified_io import read_csv

from ..utils import download_file
from .bibtex import hatemoji
from .dataset import Dataset


class HatemojiCheck(Dataset):
    def __init__(self):
        self.name = "Hatemoji Check"
        self.category = "prompts"
        self.subcategory = "statements"
        self.languages = ["English"]
        self.sources = ["Algorithmically generated"]
        self.license = "CC BY 4.0"
        self.bibtex = hatemoji
        self.alias = "hatemoji_check"
        self.splits = ["test"]
        self.generation_methods = ["templating", "algorithmic"]
        self.hazard_categories = ["hate"]

        # Other
        self.base_url = "https://github.com/HannahKirk/Hatemoji/raw/a626f63216158748d072e581efc00267031f4404/HatemojiCheck"
        self.raw_file_name = "test.csv"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{self.raw_file_name}"
        filepath = self.tmp_path / self.raw_file_name
        download_file(url, filepath)

    def load_raw_data(self) -> dict[str, list[dict]]:
        test_set = read_csv(self.tmp_path / self.raw_file_name)

        return {"test": test_set}

    def format_sample(self, x: str) -> dict:
        id = str(x["case_id"])
        label = bool(int(x["label_gold"]))
        conversation = self.format_prompt_as_conversation(x["text"])

        return dict(id=id, label=label, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [x for x in dataset["test"] if int(x["unrealistic_flags"]) == 0]
        test_set = [self.format_sample(x) for x in test_set]

        return {"test": test_set}
