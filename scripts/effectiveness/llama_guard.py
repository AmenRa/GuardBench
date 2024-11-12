import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from guardbench import benchmark


def moderate(
    conversations: list[list[dict[str, str]]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    safe_token_id: int,
    unsafe_token_id: int,
) -> list[float]:
    # Llama Guard does not support conversation starting with the assistant
    # Therefore, we drop the first utterance if it is from the assistant
    for i, x in enumerate(conversations):
        if x[0]["role"] == "assistant":
            conversations[i] = x[1:]

    # Apply chat template
    input_ids = [tokenizer.apply_chat_template(x) for x in conversations]

    # Convert input IDs to PyTorch tensor
    input_ids = torch.tensor(input_ids, device=model.device)

    # Generate output - Here is where the input moderation happens
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=0,
    )

    # Take logits for the first generated token of each input sample
    logits = output.scores[0][:, [safe_token_id, unsafe_token_id]]

    # Compute "unsafe" probabilities
    return torch.softmax(logits, dim=-1)[:, 1].tolist()


def main(device: str, datasets: list[str], batch_size: int) -> None:
    model_id = "meta-llama/LlamaGuard-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model = model.eval()

    benchmark(
        moderate=moderate,
        model_name="Llama Guard",
        batch_size=batch_size,
        datasets=datasets,
        # Moderate kwargs
        tokenizer=tokenizer,
        model=model,
        safe_token_id=tokenizer.encode("safe")[1],
        unsafe_token_id=tokenizer.encode("unsafe")[1],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str, help="Device")
    parser.add_argument("--datasets", nargs="+", default="all", help="Datasets")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    args = parser.parse_args()

    with torch.no_grad():  # Remember to avoid accumulating gradients!
        main(args.device, args.datasets, args.batch_size)
