import datasets
from unified_io import read_csv

from ..utils import download_file
from .bibtex import aart
from .dataset import Dataset


class AART(Dataset):
    def __init__(self):
        # Metadata
        self.name = "AART"
        self.category = "prompts"
        self.subcategory = "mixed"
        self.languages = ["English"]
        self.sources = ["Machine-generated"]
        self.license = "CC BY 4.0"
        self.bibtex = aart
        self.alias = "aart"
        self.splits = ["test"]
        self.generation_methods = ["llm"]
        self.hazard_categories = [
            "Violence",
            "Crime",
            "Animal Abuse",
            "Self-Harm",
            "Child",
            "Fraud",
            "Discrimination",
            "Hate",
            "Sexual Content",
        ]

        # Other
        self.repo_id = "walledai/AART"

        # THE ORIGINAL REPO HAS BEEN REMOVED
        # self.base_url = "https://github.com/google-research-datasets/aart-ai-safety-dataset/raw/fcdb2a8a3343a992cd8cbfd53eac797ac047c26f"
        # self.raw_file_name = "aart-v1-20231117.csv"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        datasets.load_dataset(self.repo_id)

        # THE ORIGINAL REPO HAS BEEN REMOVED
        # self.tmp_path.mkdir(parents=True, exist_ok=True)
        # url = f"{self.base_url}/{self.raw_file_name}"
        # filepath = self.tmp_path / self.raw_file_name
        # download_file(url, filepath)

    def load_raw_data(self) -> dict[str, list[dict]]:
        dataset = datasets.load_dataset(self.repo_id)
        test_set = dataset["train"].to_pandas().to_dict("records")

        # THE ORIGINAL REPO HAS BEEN REMOVED
        # test_set = read_csv(self.tmp_path / self.raw_file_name)
        return {"test": test_set}

    def format_sample(self, id: int, x: dict) -> dict:
        id = str(id)
        label = True
        conversation = self.format_prompt_as_conversation(x["prompt"])
        category = x["crime"]
        return dict(id=id, label=label, conversation=conversation, category=category)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]
        return {"test": test_set}
