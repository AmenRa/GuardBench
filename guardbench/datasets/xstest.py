from unified_io import read_csv

from ..utils import download_file
from .bibtex import xstest
from .dataset import Dataset


class XSTest(Dataset):
    def __init__(self):
        # Metadata
        self.name = "XSTest"
        self.category = "prompts"
        self.subcategory = "questions"
        self.languages = ["English"]
        self.sources = ["Human-generated"]
        self.license = "CC BY 4.0"
        self.bibtex = xstest
        self.alias = "xstest"
        self.splits = ["test"]
        self.generation_methods = ["human"]
        self.hazard_categories = ["generic"]
        self.purpose = ["Exagerated safety evaluation"]

        # Other
        self.base_url = "https://github.com/paul-rottger/exaggerated-safety/raw/d21c3bc28b35ed3bf54344037af9760ffae6527e"
        self.raw_file_name = "xstest_v2_prompts.csv"

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

    def compute_label(self, x: dict) -> bool:
        return "contrast" in x["type"]

    def format_sample(self, x: dict) -> dict:
        id = str(x["id_v2"])
        label = label = self.compute_label(x)
        conversation = self.format_prompt_as_conversation(x["prompt"])
        return dict(id=id, label=label, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(x) for x in dataset["test"]]
        return {"test": test_set}
