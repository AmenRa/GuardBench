import argparse

from detoxify import Detoxify

from guardbench import benchmark


def moderate(conversations: list[list[dict[str, str]]], model: Detoxify) -> list[float]:
    # Convert conversations to single texts by concatenation
    texts = ["\n".join([y["content"] for y in x]) for x in conversations]
    return model.predict(texts)["toxicity"]


def main(device: str, datasets: list[str], batch_size: int) -> None:
    # Benchmark Detoxify's Original model
    model = Detoxify("original", device=device)
    benchmark(
        moderate=moderate,
        model_name="Detoxify Original",
        batch_size=batch_size,
        datasets=datasets,
        # Moderate kwargs
        model=model,
    )

    # Benchmark Detoxify's Unbiased model
    model = Detoxify("unbiased", device=device)
    benchmark(
        moderate=moderate,
        model_name="Detoxify Unbiased",
        batch_size=batch_size,
        datasets=datasets,
        # Moderate kwargs
        model=model,
    )

    # Benchmark Detoxify's Multilingual model
    model = Detoxify("multilingual", device=device)
    benchmark(
        moderate=moderate,
        model_name="Detoxify Multilingual",
        batch_size=batch_size,
        datasets=datasets,
        # Moderate kwargs
        model=model,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str, help="Device")
    parser.add_argument("--datasets", nargs="+", default="all", help="Datasets")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size")
    args = parser.parse_args()
    main(args.device, args.datasets, args.batch_size)
