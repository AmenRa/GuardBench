# Benchmarking `Llama Guard` with `GuardBench`

Here we show how to benchmark [`Llama Guard`](https://huggingface.co/meta-llama/LlamaGuard-7b) with [`GuardBench`](https://github.com/AmenRa/guardbench).  
Note: if you want to run the following code, you first need to get [access](https://huggingface.co/meta-llama/LlamaGuard-7b) to [`Llama Guard`](https://huggingface.co/meta-llama/LlamaGuard-7b).

## Imports

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from guardbench import benchmark
```

## Load the Model and its Tokenizer

```python
model_id = "meta-llama/LlamaGuard-7b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model = model.to("cuda")
model = model.eval()
```

## Define the moderation function

The most important step is defining the function used by [`GuardBench`](https://github.com/AmenRa/guardbench) to moderate user prompts and human-AI conversations with the model under evaluation, [`Llama Guard`](https://huggingface.co/meta-llama/LlamaGuard-7b) in our case.

The `moderate` function below takes in input the `conversations` to moderate, the `tokenizer`, the `model`, and other variable needed to evaluate [`Llama Guard`](https://huggingface.co/meta-llama/LlamaGuard-7b) with our benchmark.

Please, follow the comments. 

```python
def moderate(
    conversations: list[list[dict[str, str]]],  # MANDATORY!
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

    # Apply Llama Guard's chat template to each conversation
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
```

## Run the benchmark

Once we have defined the `moderate` function, we can benchmark the model.
[`GuardBench`](https://github.com/AmenRa/guardbench) will take care of downloading and formatting the datasets and feed them to the moderation function.

```python
benchmark(
    moderate=moderate,
    model_name="Llama Guard",
    batch_size=1,
    datasets="all",
    # Moderate kwargs - the following arguments are given as input to `moderate`
    tokenizer=tokenizer,
    model=model,
    safe_token_id=tokenizer.encode("safe")[1],
    unsafe_token_id=tokenizer.encode("unsafe")[1],
)
```

## Output

Once the model has been benchmarked, results are shown as follows:

```bash
2024-06-14 at 12:42:03 | GuardBench | START    | Benchmarking Effectiveness

###
# Progress bars omitted
###

2024-06-14 at 12:49:55 | GuardBench | INFO     | Results:

| Dataset                  | F1    | Recall |
| ------------------------ | ----- | ------ |
| AART                     | 0.904 | 0.825  |
| AdvBench Behaviors       | 0.911 | 0.837  |
| AdvBench Strings         | 0.893 | 0.807  |
| BeaverTails 330k         | 0.686 | 0.546  |
| Bot-Adversarial Dialogue | 0.633 | 0.728  |
| ...                      | ...   | ...    |

2024-06-14 at 12:49:55 | GuardBench | SUCCESS  | Done

```