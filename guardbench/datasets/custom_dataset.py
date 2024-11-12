from pathlib import Path
from typing import Union

from unified_io import read_jsonl

from ..utils import write_dataset
from .dataset import Dataset


class CustomDataset(Dataset):
    def __init__(self, name: str):
        # Metadata
        self.name = name
        self.alias = name.lower().replace(" ", "_")
        self.splits = ["test"]
        self.category = None
        self.subcategory = None
        self.languages = None
        self.sources = None
        self.generation_methods = None
        self.hazard_categories = None
        self.license = None
        self.bibtex = None
        self.splits = ["test"]

        # Init super object
        super().__init__(self.alias)

    def load_data(self, path: Union[str, Path]) -> dict[str, list[dict]]:
        dataset = {"test": read_jsonl(path)}
        write_dataset(dataset, self.path)
        self.load()
        return self
