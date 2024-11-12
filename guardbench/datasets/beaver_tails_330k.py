import datasets

from .bibtex import beaver_tails
from .dataset import Dataset


class BeaverTails_330k(Dataset):
    def __init__(self):
        # Metadata
        self.name = "BeaverTails 330k"
        self.category = "conversations"
        self.subcategory = "single-turn"
        self.languages = ["English"]
        self.sources = ["Human-AI conversations / manually labelled"]
        self.license = "MIT"
        self.bibtex = beaver_tails
        self.alias = "beaver_tails_330k"
        self.splits = ["test"]
        self.generation_methods = ["real-world"]
        self.hazard_categories = [
            "Animal Abuse",
            "Child Abuse",
            "Controversial Topics, Politics",
            "Discrimination, Stereotype, Injustice",
            "Drug Abuse, Weapons, Banned Substance",
            "Financial Crime, Property Crime, Theft",
            "Hate Speech, Offensive Language",
            "Misinformation Regarding ethics, laws, and safety",
            "Non-Violent Unethical Behavior",
            "Privacy Violation",
            "Self-Harm",
            "Sexually Explicit, Adult Content",
            "Terrorism, Organized Crime",
            "Violence, Aiding and Abetting, Incitement",
        ]

        # Other
        self.repo_id = "PKU-Alignment/BeaverTails"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        datasets.load_dataset(self.repo_id)

    def combine(self, samples: list[dict]) -> dict:
        if not len(set(x["prompt"] for x in samples)) == 1:
            print(samples)
            raise ValueError("issue with prompt")
        if not len(set(x["response"] for x in samples)) == 1:
            print(samples)
            raise ValueError("issue with response")

        is_safe = sum(x["is_safe"] for x in samples) / len(samples) >= 0.5
        sample = samples[0]
        sample["is_safe"] = is_safe

        return sample

    def deduplicate(self, samples: list[dict]) -> list[dict]:
        deduplicated_samples = []

        from tqdm import tqdm

        pbar = tqdm(total=len(samples))

        while len(samples) > 0:
            x = samples[0]
            indices = [0]
            duplicates = [x]

            # Find duplicates
            for i, y in enumerate(samples[1:]):
                if x["response"] == y["response"]:
                    if x["prompt"] == y["prompt"]:
                        indices.append(i + 1)
                        duplicates.append(y)

            x = self.combine(duplicates)
            deduplicated_samples.append(x)

            for i in sorted(indices, reverse=True):
                del samples[i]

            pbar.update(len(indices))
        pbar.close()

        return deduplicated_samples

    def load_raw_data(self) -> dict[str, list[dict]]:
        dataset = datasets.load_dataset(self.repo_id)
        # train_set = dataset["330k_train"].to_pandas().to_dict("records")
        # train_set = self.deduplicate(train_set)
        test_set = dataset["330k_test"].to_pandas().to_dict("records")
        test_set = self.deduplicate(test_set)

        return {"test": test_set}

    def format_conversation(self, x: dict) -> list[dict[str, str]]:
        return [
            dict(role="user", content=x["prompt"]),
            dict(role="assistant", content=x["response"]),
        ]

    def format_sample(self, i: int, x: dict) -> dict:
        return dict(
            id=str(i),
            label=not x["is_safe"],
            conversation=self.format_conversation(x),
            target="assistant",
        )

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]

        return {"test": test_set}
