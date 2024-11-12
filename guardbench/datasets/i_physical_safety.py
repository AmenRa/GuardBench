from unified_io import read_json

from ..utils import download_file
from .bibtex import i_datasets
from .dataset import Dataset


class I_PhysicalSafety(Dataset):
    def __init__(self):
        self.name = "I-Physical-Safety"
        self.category = "prompts"
        self.subcategory = "instructions"
        self.languages = ["English"]
        self.sources = ["Human-generated"]
        self.license = "CC BY-NC 4.0"
        self.bibtex = i_datasets
        self.alias = "i_physical_safety"
        self.splits = ["test"]
        self.generation_methods = ["human"]
        self.hazard_categories = ["Physical Safety"]

        # Other
        self.base_url = "https://github.com/vinid/instruction-llms-safety-eval/raw/36a4b8d5c2177ed165bf61f59b590161394f7f12/data/evaluation"
        self.raw_file_name_safe = "I-PhysicalSafetySafe.json"
        self.raw_file_name_unsafe = "I-PhysicalSafetyUnsafe.json"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{self.raw_file_name_safe}"
        filepath = self.tmp_path / self.raw_file_name_safe
        download_file(url, filepath)

        url = f"{self.base_url}/{self.raw_file_name_unsafe}"
        filepath = self.tmp_path / self.raw_file_name_unsafe
        download_file(url, filepath)

    def load_raw_data(self) -> dict[str, list[dict]]:
        safe = read_json(self.tmp_path / self.raw_file_name_safe)["instructions"]
        unsafe = read_json(self.tmp_path / self.raw_file_name_unsafe)["instructions"]
        return {"safe": safe, "unsafe": unsafe}

    def format_sample(self, id: int, label: bool, prompt: str) -> dict:
        conversation = self.format_prompt_as_conversation(prompt)
        return dict(id=str(id), label=label, conversation=conversation)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        safe_set, unsafe_set = dataset["safe"], dataset["unsafe"]
        safe_set = [self.format_sample(i, False, x) for i, x in enumerate(safe_set)]
        unsafe_set = [
            self.format_sample(i, True, x)
            for i, x in enumerate(unsafe_set, start=len(safe_set))
        ]
        return {"test": safe_set + unsafe_set}
