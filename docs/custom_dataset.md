# Custom Dataset

To load your custom evaluation datasets, you can leverage the `CustomDataset` class.

As exemplified below, you first have to instantiate a `CustomDataset` object by providing the dataset name.
Then, you must load your evaluation data. Please, make sure that the provided samples adhere to the [`GuardBench`](https://github.com/AmenRa/guardbench)'s [data format](data_format.md) and the dataset is saved in a [JSONl](https://jsonlines.org/) file.
Finally, you can employ [`GuardBench`](https://github.com/AmenRa/guardbench)'s benchmarking pipeline to evaluate moderation functions on your custom dataset.

```python
from guardbench import CustomDataset

# Instantiate
dataset_name = "my_custom_dataset"
custom_dataset = CustomDataset(dataset_name)

# Load data - data will be stored for future usage
custom_dataset.load_data("path/to/dataset.jsonl")

# Evaluation
benchmark(moderate, datasets=[dataset_name])
```