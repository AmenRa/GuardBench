# Data format

Here, we describe how [`GuardBench`](https://github.com/AmenRa/guardbench) format the datasets' samples before feeding them to the model under evaluation, i.e., passing them to the provided moderation function.

Both user prompts and human-AI conversations are formatted with the layout shown below.

```python
prompt = [
    {"role": "user", "content": "What is GuardBench?"}
]
```

```python
conversation = [
    {"role": "user", "content": "What is GuardBench?"},
    {
        "role": "assistant",
        "content": "GuardBench is a Python library for guardrail models evaluation.",
    },
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...."},
    ...
]
```