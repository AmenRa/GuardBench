from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
from unified_io import create_path, write_json

from ..datasets import DATASETS, load_dataset
from ..evaluate import evaluate


def get_results_table(results) -> None:
    headers = ["Dataset", "F1", "Recall"]
    return tabulate(results, headers=headers, tablefmt="github", disable_numparse=True)


def collate_fn(batch):
    id = [x["id"] for x in batch]
    label = [x["label"] for x in batch]
    conversation = [x["conversation"] for x in batch]
    return id, label, conversation


def benchmark(
    moderate: callable,
    model_name: str = "moderator",
    out_dir: str = "results",
    batch_size: int = 32,
    datasets: list = "all",
    **kwargs,
) -> None:
    """Benchmark the effectiveness of the provided moderation function/model.     Additional keyword arguments are passed to the provided moderation function. For example, you can pass the tokenizer and the model to the moderation function. Check the official repository for examples and tutorials.

    Args:
        moderate (callable): Moderation function. It must have at least one parameter named `conversations`.
        model_name (str, optional): "Name of the moderation model". Defaults to "moderator".
        out_dir (str, optional): Directory for the moderation outputs. Defaults to "results".
        batch_size (int, optional): Batch size. Defaults to 32.
        datasets (list, optional): Datasets selected for evaluation. Defaults to "all".
    """

    # Set datasets if None  ----------------------------------------------------
    if datasets == "all":
        datasets = list(DATASETS)

    # Benchmarking Effectiveness -----------------------------------------------
    logger.start("Benchmarking Effectiveness")
    results = []
    for i, dataset_name in enumerate(datasets):
        # Dataset --------------------------------------------------------------
        dataset = load_dataset(dataset_name)
        batch_generator = dataset.generate_batches(batch_size)

        # Inference ------------------------------------------------------------
        idx = str(i + 1).zfill(2) if len(datasets) > 9 else i + 1
        desc = f"{idx}/{len(datasets)} - {dataset.name}"
        tqdm_kwargs = dict(desc=desc, dynamic_ncols=True, mininterval=1.0)

        ids, y_true, y_pred_prob = [], [], []

        for batch in tqdm(list(batch_generator), **tqdm_kwargs):
            batch_ids, batch_y_true, batch_conversations = collate_fn(batch)
            ids += batch_ids
            y_true += batch_y_true
            y_pred_prob += moderate(conversations=batch_conversations, **kwargs)

        y_true = dict(zip(ids, y_true))
        y_pred_prob = dict(zip(ids, y_pred_prob))

        # Save predictions -----------------------------------------------------
        pred_path = create_path(out_dir)
        write_json(y_pred_prob, pred_path / dataset_name / f"{model_name}.json")

        # Evaluate -------------------------------------------------------------
        report = evaluate(y_true, y_pred_prob)
        results.append(
            [
                dataset.name,
                "{:.3f}".format(round(report["f1"], 3)),
                "{:.3f}".format(round(report["recall"], 3)),
            ]
        )

    results_table = get_results_table(results)
    logger.info(f"Results:\n{results_table}")

    logger.success("Done")
