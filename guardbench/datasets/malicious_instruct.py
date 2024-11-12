from unified_io import read_list

from ..utils import download_file
from .bibtex import malicious_instruct
from .dataset import Dataset


class MaliciousInstruct(Dataset):
    def __init__(self):
        # Metadata
        self.name = "MaliciousInstruct"
        self.category = "prompts"
        self.subcategory = "instructions"
        self.languages = ["English"]
        self.sources = ["Machine-generated"]
        self.license = "MIT"
        self.bibtex = malicious_instruct
        self.alias = "malicious_instruct"
        self.splits = ["test"]
        self.generation_methods = ["machine", "llm"]
        self.hazard_categories = [
            "sychological manipulation",
            "sabotage",
            "theft",
            "defamation",
            "cyberbullying",
            "false accusation",
            "tax",
            "fraud",
            "hacking",
            "fraud",
            "illegal drug use",
        ]

        # Other
        self.base_url = "https://github.com/Princeton-SysML/Jailbreak_LLM/raw/1e490531c0ef4bc13430178bf1fbc8f813e34b1a/data"
        self.raw_file_name = "MaliciousInstruct.txt"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{self.raw_file_name}"
        filepath = self.tmp_path / self.raw_file_name
        download_file(url, filepath)

    def load_raw_data(self) -> dict[str, list[dict]]:
        test_set = read_list(self.tmp_path / self.raw_file_name)
        return {"test": test_set}

    def format_sample(self, i: int, prompt: str) -> dict:
        conversation = self.format_prompt_as_conversation(prompt)
        return dict(id=str(i), label=True, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]
        return {"test": test_set}
