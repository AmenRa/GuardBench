from huggingface_hub import hf_hub_download
from unified_io import read_csv

from .bibtex import hex_phi
from .dataset import Dataset


class HEx_PHI(Dataset):
    def __init__(self):
        self.name = "HEx-PHI"
        self.category = "prompts"
        self.subcategory = "questions"
        self.languages = ["English"]
        self.sources = ["Human-generated"]
        self.license = "https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI#hex-phi-dataset-license-agreement"
        self.bibtex = hex_phi
        self.alias = "hex_phi"
        self.splits = ["test"]
        self.generation_methods = ["human"]
        self.hazard_categories = [
            "Illegal Activity",
            # "Child Abuse Content",  # REMOVED BY AUTHORS
            "Hate Harass Violence",
            "Malware",
            "Physical Harm",
            "Economic Harm",
            "Fraud Deception",
            "Adult Content",
            "Political Campaigning",
            "Privacy Violation Activity",
            "Tailored Financial Advice",
        ]

        # Other
        self.repo_id = "LLM-Tuning-Safety/HEx-PHI"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)

        for i in range(1, 12):
            if i == 2:  # REMOVED BY AUTHORS
                continue

            hf_hub_download(
                repo_id=self.repo_id,
                filename=f"category_{i}.csv",
                repo_type="dataset",
                local_dir=self.tmp_path,
            )

    def load_raw_data(self) -> dict[str, list[dict]]:
        kwargs = {"fieldnames": ["text"]}
        dataset = []

        data = read_csv(self.tmp_path / "category_1.csv", kwargs=kwargs)
        data = [{"text": x["text"], "category": "Illegal Activity"} for x in data]
        dataset += data

        # REMOVED BY AUTHORS
        # data = read_csv(self.tmp_path / "category_2.csv", kwargs=kwargs)
        # data = [{"text": x["text"], "category": "Child Abuse Content"} for x in data]
        # dataset += data

        data = read_csv(self.tmp_path / "category_3.csv", kwargs=kwargs)
        data = [{"text": x["text"], "category": "Hate Harass Violence"} for x in data]
        dataset += data

        data = read_csv(self.tmp_path / "category_4.csv", kwargs=kwargs)
        data = [{"text": x["text"], "category": "Malware"} for x in data]
        dataset += data

        data = read_csv(self.tmp_path / "category_5.csv", kwargs=kwargs)
        data = [{"text": x["text"], "category": "Physical Harm"} for x in data]
        dataset += data

        data = read_csv(self.tmp_path / "category_6.csv", kwargs=kwargs)
        data = [{"text": x["text"], "category": "Economic Harm"} for x in data]
        dataset += data

        data = read_csv(self.tmp_path / "category_7.csv", kwargs=kwargs)
        data = [{"text": x["text"], "category": "Fraud Deception"} for x in data]
        dataset += data

        data = read_csv(self.tmp_path / "category_8.csv", kwargs=kwargs)
        data = [{"text": x["text"], "category": "Adult Content"} for x in data]
        dataset += data

        data = read_csv(self.tmp_path / "category_9.csv", kwargs=kwargs)
        data = [{"text": x["text"], "category": "Political Campaigning"} for x in data]
        dataset += data

        data = read_csv(self.tmp_path / "category_10.csv", kwargs=kwargs)
        data = [
            {"text": x["text"], "category": "Privacy Violation Activity"} for x in data
        ]
        dataset += data

        data = read_csv(self.tmp_path / "category_11.csv", kwargs=kwargs)
        data = [
            {"text": x["text"], "category": "Tailored Financial Advice"} for x in data
        ]
        dataset += data

        return {"test": dataset}

    def format_sample(self, i: int, x: dict) -> dict:
        id = str(i)
        label = True
        conversation = self.format_prompt_as_conversation(x["text"])
        category = x["category"]

        return dict(id=id, label=label, conversation=conversation, category=category)

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(i, x) for i, x in enumerate(dataset["test"])]

        return {"test": test_set}
