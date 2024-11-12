import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from guardbench import benchmark


def moderate(
    conversations: list[list[dict[str, str]]],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    safe_token_id: int,
    unsafe_token_id: int,
) -> list[float]:
    # Convert conversations to single texts by concatenation
    texts = ["\n".join([y["content"] for y in x]) for x in conversations]

    # Apply prompt template
    texts = ["ToxicChat: " + x for x in texts]

    # Tokenize texts
    input = tokenizer(
        texts,
        max_length=4096,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Move input to model device
    input = {k: v.to(model.device) for k, v in input.items()}

    # Generate output
    output = model.generate(
        **input,
        max_new_tokens=1,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # Take logits for the first generated token of each prompt
    logits = output.scores[0][:, [safe_token_id, unsafe_token_id]]

    # Compute "unsafe" probabilities
    return torch.softmax(logits, dim=-1)[:, 1].tolist()


def main(device: str, datasets: list[str], batch_size: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained("t5-large", model_max_length=4096)

    model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/toxicchat-t5-large-v1.0")
    model = model.to(device)
    model = model.eval()

    benchmark(
        moderate=moderate,
        model_name="Toxic Chat T5",
        batch_size=batch_size,
        datasets=datasets,
        # Moderate kwargs
        tokenizer=tokenizer,
        model=model,
        safe_token_id=tokenizer.encode("negative")[0],
        unsafe_token_id=tokenizer.encode("positive")[0],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str, help="Device")
    parser.add_argument("--datasets", nargs="+", default="all", help="Datasets")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    args = parser.parse_args()

    with torch.no_grad():
        main(args.device, args.datasets, args.batch_size)
