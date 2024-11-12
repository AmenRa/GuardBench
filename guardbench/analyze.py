from unified_io import read_json

from .datasets import load_dataset


def analyze(model_name: str, dataset_name: str, out_dir: str = "results"):
    dataset = load_dataset(dataset_name)
    prediction = read_json(f"{out_dir}/{dataset_name}/{model_name}.json")

    safe_as_unsafe = []
    for x in dataset:
        if not x["label"] and prediction[x["id"]] > 0.5:
            safe_as_unsafe.append(x | {"prediction": prediction[x["id"]]})

    unsafe_as_safe = []
    for x in dataset:
        if x["label"] and prediction[x["id"]] <= 0.5:
            unsafe_as_safe.append(x | {"prediction": prediction[x["id"]]})

    return {"safe_as_unsafe": safe_as_unsafe, "unsafe_as_safe": unsafe_as_safe}
