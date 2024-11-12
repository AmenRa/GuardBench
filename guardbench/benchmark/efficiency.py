from secrets import randbelow
from time import time

from loguru import logger
from unified_io import create_path, write_json

from ..datasets import DATASETS, load_dataset


def benchmark(
    moderate: callable,
    model_name: str = "moderator",
    out_dir: str = "results",
    datasets: list = None,
    **kwargs,
) -> None:
    if datasets is None:
        datasets = list(DATASETS)

    # Benchmarking Efficiency --------------------------------------------------
    logger.start("Benchmarking Efficiency")
    total_time = 0.0
    n_queries = 0
    for dataset_name in datasets:
        # Dataset --------------------------------------------------------------
        dataset = load_dataset(dataset_name)

        # Inference ------------------------------------------------------------
        start_time = time()
        for _ in range(100):
            sample = dataset[randbelow(len(dataset))]
            moderate(conversations=sample["conversation"], **kwargs)
        total_time += time() - start_time

    qps = n_queries / total_time

    # Save throughput ----------------------------------------------------------
    pred_path = create_path(out_dir)
    write_json({"throughput": qps}, pred_path / "throughput" / f"{model_name}.json")

    logger.info(f"Throughput (q/s): {round(qps, 2)}")

    logger.success("Done")
