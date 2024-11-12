from unified_io import read_csv

from ..utils import download_file
from .bibtex import do_anything_now
from .dataset import Dataset


class DoAnythingNow_Questions(Dataset):
    def __init__(self):
        # Metadata
        self.name = "Do Anything Now Questions"
        self.category = "prompts"
        self.subcategory = "questions"
        self.languages = ["English"]
        self.sources = [
            "Discord",
            "Reddit",
            "AIPRM",
            "FlowGPT",
            "JailbreakChat",
            "Awesome ChatGPT Prompts",
            "OCR-Prompts",
        ]
        self.license = "MIT"
        self.bibtex = do_anything_now
        self.alias = "do_anything_now_questions"
        self.splits = ["test"]
        self.generation_methods = ["machine", "llm"]
        self.hazard_categories = [
            "Illegal Activity",
            "Hate Speech",
            "Malware Generation",
            "Physical Harm",
            "Fraud",
            "Pornography",
            "Political Lobbying",
            "Privacy Violence",
            "Legal Opinion",
            "Financial Advice",
            "Health Consultation",
            "Government Decision",
        ]

        # Other
        self.base_url = "https://github.com/verazuo/jailbreak_llms/raw/09ab5ecb9632087119af25a04b88b0f496c989be/data/forbidden_question"
        self.raw_file_name = "forbidden_question_set.csv"

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

    def format_sample(self, id: int, x: dict) -> dict:
        conversation = self.format_prompt_as_conversation(x["question"])
        return dict(id=str(id), label=True, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]
        return {"test": test_set}
