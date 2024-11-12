from loguru import logger
from tabulate import tabulate
from unified_io import read_jsonl

from ..paths import datasets_path, tmp_path
from ..utils import write_dataset


class Dataset:
    def __init__(self, alias):
        self.path = datasets_path() / alias.replace("_", "-")
        self.tmp_path = tmp_path() / alias.replace("_", "-")
        self.dataset = None
        self.current_split = None

    def load(self, split: str = "test"):
        """Load the specified dataset split.

        Args:
            split (str, optional): Split to load. Defaults to "test".

        Raises:
            Exception: Raise exception if the split is not available.

        Returns:
            Dataset: Returns self.
        """
        if split in self.splits:
            if not (self.path / f"{split}.jsonl").exists():
                self.download_and_format()

            self.dataset = read_jsonl(self.path / f"{split}.jsonl")
            self.index = {x["id"]: i for i, x in enumerate(self.dataset)}
            self.current_split = split
            self.set_metadata()
            return self
        else:
            raise Exception(f"{self.name} does not have a {split} set")

    def get(self, id: str) -> dict:
        """Get sample by ID.

        Args:
            id (str): ID of the sample.

        Returns:
            dict: Dataset sample.
        """
        idx = self.index[id]
        return self.dataset[idx]

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        raise NotImplementedError

    def load_raw_data(self):
        raise NotImplementedError

    def format_prompt_as_conversation(self, prompt: str) -> list[dict[str, str]]:
        """Format the given prompt in conversation format.

        Args:
            prompt (str): Prompt to format as conversation.

        Returns:
            list[dict[str, str]]: Conversation.
        """
        return [dict(role="user", content=prompt.strip())]

    def format_prompt_response_as_conversation(
        self, prompt: str, response: str
    ) -> list[dict[str, str]]:
        """Format the given prompt and response in conversation format.

        Args:
            prompt (str): Prompt to include in conversation.
            response (str): Response to include in conversation.

        Returns:
            list[dict[str, str]]: Conversation.
        """
        return [
            dict(role="user", content=prompt),
            dict(role="assistant", content=response),
        ]

    def format_sample(self):
        raise NotImplementedError

    def format_data():
        raise NotImplementedError

    def download_and_format(self):
        """Download and format dataset."""
        logger.info(f"Downloading {self.name}")
        self.path.mkdir(parents=True, exist_ok=True)
        self.download_raw_data()
        dataset = self.load_raw_data()
        dataset = self.format_data(dataset)
        write_dataset(dataset, self.path)

    def get_ground_truth(self, split: str = "test") -> dict[str, bool]:
        """Load the ground truth of the specified split.

        Args:
            split (str, optional): Split for which to load the ground truth. Defaults to "test".

        Returns:
            dict[str, bool]: Ground truth.
        """
        if self.current_split != split:
            self.load(split)
        return {x["id"]: x["label"] for x in self.dataset}

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Compute dataset statistics.

        Returns:
            dict[str, dict[str, int]]: Dataset statistics.
        """
        stats = {}
        for split in ["test"]:
            try:
                if self.current_split != split:
                    self.load(split)
                total = len(self.dataset)
                unsafe = sum([x["label"] for x in self.dataset])
            except Exception:
                total, unsafe = 0, 0
            stats[split] = {"total": total, "unsafe": unsafe}
        return stats

    def set_metadata(self):
        """Set dataset metadata."""
        stats = self.get_stats()
        self.metadata = {
            "name": self.name,
            "category": self.category,
            "subcategory": self.subcategory,
            "languages": self.languages,
            "sources": self.sources,
            "generation_methods": self.generation_methods,
            "hazard_categories": self.hazard_categories,
            "license": self.license,
            "bibtex": self.bibtex,
            "alias": self.alias,
            "splits": self.splits,
            "test": stats["test"],
        }

    def print_stats(self):
        """Print dataset statistics."""
        headers = ["split", "total", "unsafe"]
        data = self.get_stats()
        data = [[k, *list(v.values())] for k, v in data.items()]
        print(tabulate(tabular_data=data, headers=headers, tablefmt="github"))

    def generate_batches(self, batch_size: int, shuffle: bool = False):
        """Batch generator.

        Args:
            batch_size (int): Batch size.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.

        Yields:
            List[Union[str, Dict, np.ndarray]]: Batch of items.
        """
        indices = list(range(len(self.dataset)))

        if shuffle:
            shuffle(indices)

        for i in range(0, len(self.dataset), batch_size):
            batch_indices = indices[i : i + batch_size]
            yield [self.dataset[i] for i in batch_indices]

    def __getitem__(self, index: int) -> str:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def __next__(self):
        return self.dataset

    def __iter__(self):
        self.iteration_index = 0
        return self

    def __next__(self):
        if hasattr(self, "iteration_index") == False:
            self.iteration_index = 0
        if self.iteration_index < len(self.dataset):
            result = self.dataset[self.iteration_index]
            self.iteration_index += 1
            return result
        else:
            raise StopIteration
