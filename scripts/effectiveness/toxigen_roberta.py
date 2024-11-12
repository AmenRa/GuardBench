import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from guardbench import benchmark


def moderate(
    conversations: list[list[dict[str, str]]],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
) -> list[float]:
    # Convert conversations to single texts by concatenation
    texts = ["\n".join([y["content"] for y in x]) for x in conversations]

    # Tokenize texts
    input = tokenizer(
        texts,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Move input to model device
    input = {k: v.to(model.device) for k, v in input.items()}

    # Compute logits
    logits = model(**input).logits

    # Compute "unsafe" probabilities
    return torch.softmax(logits, dim=-1)[:, 1].tolist()


def main(device: str, datasets: list[str], batch_size: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    model = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_roberta")
    model = model.to(device)
    model = model.eval()

    benchmark(
        moderate=moderate,
        model_name="ToxiGen RoBERTa",
        batch_size=batch_size,
        datasets=datasets,
        # Moderate kwargs
        model=model,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str, help="Device")
    parser.add_argument("--datasets", nargs="+", default="all", help="Datasets")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size")
    args = parser.parse_args()

    with torch.no_grad():
        main(args.device, args.datasets, args.batch_size)
