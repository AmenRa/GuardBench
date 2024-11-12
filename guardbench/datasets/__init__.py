from importlib import import_module

from tabulate import tabulate

from ..paths import tmp_path
from ..utils import remove_raw_data
from .custom_dataset import CustomDataset

DATASETS = {
    "aart": "AART",
    "advbench_behaviors": "AdvBench Behaviors",
    "advbench_strings": "AdvBench Strings",
    "beaver_tails_330k": "BeaverTails 330k",
    "bot_adversarial_dialogue": "Bot-Adversarial Dialogue",
    "cat_qa": "CatQA",
    "convabuse": "ConvAbuse",
    "decoding_trust_stereotypes": "DecodingTrust Stereotypes",
    "dices_350": "DICES 350",
    "dices_990": "DICES 990",
    "do_anything_now_questions": "Do Anything Now Questions",
    "do_not_answer": "DoNotAnswer",
    "dynahate": "DynaHate",
    "harmbench_behaviors": "HarmBench Behaviors",
    "harmful_q": "HarmfulQ",
    "harmful_qa_questions": "HarmfulQA Questions",
    "harmful_qa": "HarmfulQA",
    "hatecheck": "HateCheck",
    "hatemoji_check": "Hatemoji Check",
    "hex_phi": "HEx-PHI",
    "i_cona": "I-CoNa",
    "i_controversial": "I-Controversial",
    "i_malicious_instructions": "I-MaliciousInstructions",
    "i_physical_safety": "I-Physical-Safety",
    "malicious_instruct": "MaliciousInstruct",
    "mitre": "MITRE",
    "openai_moderation_dataset": "OpenAI Moderation Dataset",
    "prosocial_dialog": "ProsocialDialog",
    "safe_text": "SafeText",
    "simple_safety_tests": "SimpleSafetyTests",
    "strong_reject_instructions": "StrongREJECT Instructions",
    "tdc_red_teaming": "TDCRedTeaming",
    "toxic_chat": "Toxic Chat",
    "toxigen": "ToxiGen",
    "xstest": "XSTest",
}

UNSAFE_ONLY_DATASETS = [
    "AART",
    "AdvBench Behaviors",
    "AdvBench Strings",
    "CatQA",
    "DangerousQA",
    "DecodingTrust Stereotypes",
    "Do Anything Now Questions",
    "Do Anything Now",
    "DoNotAnswer",
    "HarmBench Behaviors",
    "HarmfulQA Questions",
    "HEx-PHI",
    "I-CoNa",
    "I-Controversial",
    "I-MaliciousInstructions",
    "MaliciousInstruct",
    "MITRE",
    "SafetyInstructions",
    "SimpleSafetyTests",
    "StrongREJECT Instructions",
    "TDCRedTeaming",
    "Toxic Chat Jailbreaks",
]


def get_dataset(dataset: str):
    """Get the specified dataset Python class.

    Args:
        dataset (str): Dataset to get.
    """
    if dataset in DATASETS:
        module = import_module(f"guardbench.datasets.{dataset}")
        class_name = [x for x in dir(module) if x != "Dataset"][0]
        return getattr(module, class_name)()

    else:
        custom_dataset = CustomDataset(dataset)
        if not (custom_dataset.path / "test.jsonl").exists():
            raise FileNotFoundError(f"Dataset `{dataset}` not found.")
        return custom_dataset


def load_dataset(dataset: str):
    """Load the specified dataset.

    Args:
        dataset (str): Dataset to load.
    """
    return get_dataset(dataset).load()


def load_datasets(datasets: list[str]):
    """Load the specified datasets.

    Args:
        datasets (list[str]): Datasets to load.
    """
    return [load_dataset(dataset) for dataset in datasets]


def download_dataset(dataset: str):
    """Download the specified dataset.

    Args:
        dataset (str): Dataset to download.
    """
    return get_dataset(dataset).download_and_format()


def download(datasets: list[str] = "all"):
    """Download the specified datasets.

    Args:
        datasets (list[str], optional): Datasets to download. Defaults to "all".
    """
    if datasets == "all":
        datasets = list(DATASETS)

    for dataset in datasets:
        download_dataset(dataset)
        remove_raw_data(tmp_path())


def download_all():
    """Download all datasets."""
    download()


def list_datasets():
    """List dataset aliases."""
    for x in DATASETS.values():
        print(x)


def get_datasets_by(
    category: str = None, subcategory: str = None, language: str = None
) -> list[str]:
    """Returns a list of dataset aliases that fulfill the provided selection criteria.

    Args:
        category (str): Category.
        subcategory (str, optional): Sub-category. Defaults to None.
        language (str, optional): Language. Defaults to None.

    Returns:
        list[str]: List of datasets.
    """
    datasets = []

    for name in DATASETS:
        dataset = get_dataset(name)

        category_flag = True
        subcategory_flag = True
        language_flag = True

        if category is not None and dataset.category != category:
            category_flag = False

        if subcategory is not None and dataset.subcategory != subcategory:
            subcategory_flag = False

        if language is not None and language.lower() not in set(
            lang.lower() for lang in dataset.languages
        ):
            language_flag = False

        if category_flag and subcategory_flag and language_flag:
            datasets.append(name)

    return datasets


def get_sorted_dataset_list() -> list:
    """Returns the list of dataset sorted as in the paper.

    Returns:
        list: List of datasets.
    """
    return [
        # Prompts - Instructions
        *get_datasets_by("prompts", "instructions", "english"),
        # Prompts - Questions
        *get_datasets_by("prompts", "questions", "english"),
        # Prompts - Statements
        *get_datasets_by("prompts", "statements", "english"),
        # Prompts - Mixed
        *get_datasets_by("prompts", "mixed", "english"),
        # Conversation - Single-Turn
        *get_datasets_by("conversations", "single-turn", "english"),
        # Conversation - Multi-Turn
        *get_datasets_by("conversations", "multi-turn", "english"),
        # Multi-Lang
        *get_datasets_by("prompts", "mixed", "german"),
        *get_datasets_by("prompts", "mixed", "french"),
        *get_datasets_by("prompts", "mixed", "italian"),
        *get_datasets_by("prompts", "mixed", "spanish"),
    ]


def export_latex_table():
    headers = [
        "#",
        "Dataset",
        "Category",
        "Sub-category",
        "Total",
        "Unsafe",
        "Generation",
        "License",
        "Citation",
    ]

    tabular_data = []

    datasets = get_sorted_dataset_list()
    datasets = load_datasets(datasets)

    for dataset in datasets:
        tabular_data += [
            [
                "\\rownumber",
                dataset.metadata["name"],
                dataset.metadata["category"].title(),
                dataset.metadata["subcategory"].title(),
                dataset.metadata["test"]["total"],
                round(
                    dataset.metadata["test"]["unsafe"]
                    / dataset.metadata["test"]["total"]
                    * 100
                ),
                " ".join(dataset.metadata["generation_methods"]),
                dataset.metadata["license"],
                "\cite{" + dataset.metadata["bibtex"].split("{")[1].split(",")[0] + "}",
            ]
        ]

    table = tabulate(
        tabular_data=tabular_data,
        headers=headers,
        tablefmt="latex_raw",
        disable_numparse=True,
    )

    print(table)


def export_markdown_table():
    def format_name_col(name: str) -> str:
        return "[" + name + "]" + "(#" + name.lower().replace(" ", "-") + ")"

    def format_unsafe_col(unsafe: int, total: int) -> str:
        return str(round(unsafe / total * 100)) + "%"

    headers = [
        "Dataset",
        "Category",
        "Sub-category",
        "Total",
        "Unsafe",
        "Alias",
    ]

    tabular_data = []

    datasets = get_sorted_dataset_list()
    datasets = load_datasets(datasets)

    for dataset in datasets:
        tabular_data += [
            [
                format_name_col(dataset.metadata["name"]),
                dataset.metadata["category"].title(),
                dataset.metadata["subcategory"].title(),
                dataset.metadata["test"]["total"],
                format_unsafe_col(
                    dataset.metadata["test"]["unsafe"],
                    dataset.metadata["test"]["total"],
                ),
                dataset.metadata["alias"],
            ]
        ]

    table = tabulate(
        tabular_data=tabular_data,
        headers=headers,
        tablefmt="pipe",
        # disable_numparse=True,
    )

    print(table)


def export_datasets_doc():
    template = [
        "## {name}",
        "",
        "- Language(s): {languages}",
        "- Source(s): {sources}",
        "- License: `{license}`",
        "- Alias: `{alias}`",
        "",
        "<details>",
        "  <summary>BibTeX</summary>",
        "",
        "  ```bibtex",
        "  {bibtex}",
        "  ```",
        "</details>",
    ]
    template = "\n".join(template)

    docs = []

    datasets = get_sorted_dataset_list()
    datasets = load_datasets(datasets)

    for dataset in datasets:
        docs.append(
            template.format(
                name=dataset.metadata["name"],
                languages=", ".join(dataset.metadata["languages"]),
                sources=", ".join(dataset.metadata["sources"]),
                license=dataset.metadata["license"],
                alias=dataset.metadata["alias"],
                bibtex=dataset.metadata["bibtex"].replace("\n", "\n  "),
            )
        )

    print("\n\n".join(docs))
