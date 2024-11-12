from pathlib import Path
from typing import Union

import numpy as np
from IPython.display import Markdown, display
from tabulate import tabulate
from unified_io import read_json

from .datasets import DATASETS, UNSAFE_ONLY_DATASETS, load_dataset
from .evaluate import evaluate


class Report:
    def __init__(
        self,
        models: Union[list[str], list[dict[str, str]]],
        datasets: list[str] = "all",
        out_dir: str = "results",
    ):
        """Evaluation report.

        Args:
            models (Union[list[str], list[dict[str, str]]]): Models to compare.
            datasets (list[str], optional): Datasets to consider. Defaults to "all".
            out_dir (str, optional): Directory where results are saved. Defaults to "results".
        """
        if datasets == "all":
            datasets = list(DATASETS)

        self.models = self.extract_model_names(models)
        self.alias_map = self.build_alias_map(models)
        self.datasets = datasets
        self.out_dir = Path(out_dir)

        self.load_data()

    def build_alias_map(
        self, models: Union[list[str], list[dict[str, str]]]
    ) -> dict[str, str]:
        if isinstance(models[0], str):
            return {x: x for x in models}
        return {x["name"]: x["alias"] for x in models}

    def extract_model_names(
        self, models: Union[list[str], list[dict[str, str]]]
    ) -> list[str]:
        if isinstance(models[0], str):
            return models
        return [x["name"] for x in models]

    def load_data(self):
        self.report = {}

        for dataset_code_name in self.datasets:
            dataset_name = DATASETS.get(dataset_code_name, False)
            if dataset_name is False:
                continue

            self.report[dataset_name] = {}

            y_true = load_dataset(dataset_code_name).get_ground_truth("test")

            for model_name in self.models:
                y_pred_prob = read_json(
                    self.out_dir / f"{dataset_code_name}/{model_name}.json"
                )
                self.report[dataset_name][model_name] = evaluate(y_true, y_pred_prob)

    def print_tables(self, metrics: list[str] = None):
        if metrics is None:
            metrics = [
                "precision",
                "recall",
                "f1",
                "auprc",
                "sensitivity",
                "specificity",
                "g_mean",
                "fpr",
                "fnr",
                "tp",
                "tn",
                "fp",
                "fn",
            ]

        metrics_map = {
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "auprc": "AUPRC",
            "sensitivity": "Sensitivity",
            "specificity": "Specificity",
            "g_mean": "G-Mean",
            "fpr": "FPR",
            "fnr": "FNR",
            "tp": "TP",
            "tn": "TN",
            "fp": "FP",
            "fn": "FN",
        }
        headers = ["Model"] + [metrics_map[k] for k in metrics]

        for dataset_name, results in self.report.items():
            print(f"### {dataset_name}")

            tabular_data = []
            for model_name, model_results in results.items():
                row = [self.alias_map[model_name]]
                for metric in metrics:
                    score = model_results[metric]
                    if isinstance(score, float):
                        score = "{:.3f}".format(round(score, 3))
                    row.append(score)
                tabular_data.append(row)

            table = tabulate(
                tabular_data=tabular_data,
                headers=headers,
                tablefmt="github",
                disable_numparse=True,
            )

            print(table, "\n")

    def to_table(self, metric: str = "f1"):
        headers = ["Dataset"] + ["Metric"] + [self.alias_map[m] for m in self.models]

        tabular_data = []
        for dataset_name, results in self.report.items():
            current_metric = (
                "recall" if dataset_name in UNSAFE_ONLY_DATASETS else metric
            )
            # current_metric = metric
            row = [dataset_name, current_metric.capitalize()] + [
                results[m][current_metric] for m in self.models
            ]
            tabular_data.append(row)

        wins = [0] * len(self.models)
        for i in range(len(self.models)):
            for row in tabular_data:
                if np.argmax(row[2:]) == i:
                    wins[i] += 1
        tabular_data.append(["Wins", ""] + wins)

        for i, row in enumerate(tabular_data):
            for j, x in enumerate(row):
                if isinstance(x, float):
                    tabular_data[i][j] = "{:.3f}".format(round(x, 3))

        return headers, tabular_data

    def to_markdown(self, metric: str = "f1"):
        headers, tabular_data = self.to_table(metric)
        return tabulate(
            tabular_data=tabular_data,
            headers=headers,
            tablefmt="github",
            disable_numparse=True,
        )

    def to_latex(self, metric: str = "f1", highlight: bool = True):
        """Convert report to LaTeX table.

        Args:
            metric (str, optional): _description_. Defaults to "f1".
            highlight (bool, optional): If True, best results are in boldface and second-best results are underlined. Defaults to True.
        """
        headers, tabular_data = self.to_table(metric)

        if highlight:
            for i in range(len(tabular_data)):
                row = tabular_data[i]
                scores = row[2:]
                indices = np.argsort([float(x) for x in scores])[::-1]

                best_score = scores[indices[0]]
                best_indices = [j - 2 for j in range(len(row)) if row[j] == best_score]

                second_best_score = scores[indices[len(best_indices)]]
                second_best_indices = [
                    j - 2 for j in range(len(row)) if row[j] == second_best_score
                ]

                for j in best_indices:
                    scores[j] = f"$   \\textbf{{{scores[j]}}}$"
                for j in second_best_indices:
                    scores[j] = f"$\\underline{{{scores[j]}}}$"
                for j in range(len(scores)):
                    if j not in best_indices and j not in second_best_indices:
                        scores[j] = "$" + " " * 11 + f"{scores[j]}" + " " * 1 + "$"

                tabular_data[i] = row[:2] + scores

        table = tabulate(
            tabular_data=tabular_data,
            headers=headers,
            tablefmt="latex",
            disable_numparse=True,
        )

        rows = table.split("\n")
        rows.insert(-3, "\\hline")

        if highlight:
            rows[2] = rows[2].split("&")
            rows[2] = (
                rows[2][:2]
                + [x[:-2] for x in rows[2][2:-1]]
                + [rows[2][-1][:-4] + "\\\\"]
            )
            rows[2] = "&".join(rows[2])

            for i, row in enumerate(rows):
                row = row.replace("\\textbackslash{}", "\\")
                row = row.replace("\\{", "{")
                row = row.replace("\\}", "}")
                row = row.replace("\\$", "")
                # row = row.replace("}  &", "} &")
                # row = row.replace("}  \\\\", "} \\\\")
                rows[i] = row

        table = ("\n").join(rows)

        prefix = "\n".join(
            [
                "\\begin{table*}[!ht]",
                "\\centering",
            ]
        )

        suffix = "\n".join(
            [
                (
                    "\\caption{Evaluation results. Best results are highlighted in boldface. Second-best results are underlined.}"
                    if highlight
                    else "\\caption{Evaluation results.}"
                ),
                "\\label{tab:results}",
                "\\end{table*}",
            ]
        )

        table = "\n".join([prefix, table, suffix])

        print(table)

    def display(self, metric: str = "f1"):
        """Display the evaluation report. Best results are highlighted in bold. Second-best results are underlined.

        Args:
            metric (str, optional): Metric to use for results evaluation. Defaults to "f1".
        """

        table = self.to_markdown(metric)

        # Add highlighting
        rows = table.split("\n")[:-1]
        for i in range(2, len(rows)):
            row = rows[i].split("|")
            scores = row[3:-1]
            indices = np.argsort([float(x) for x in scores])[::-1]

            best_score = scores[indices[0]]
            best_indices = [j - 3 for j in range(len(row)) if row[j] == best_score]

            second_best_score = scores[indices[len(best_indices)]]
            second_best_indices = [
                j - 3 for j in range(len(row)) if row[j] == second_best_score
            ]

            for j in best_indices:
                scores[j] = f"**{scores[j].strip()}**"
            for j in second_best_indices:
                scores[j] = f"<ins>{scores[j].strip()}</ins>"

            rows[i] = ("|").join(row[:3] + scores + row[-1:])

        table = ("\n").join(rows + [table.split("\n")[-1]])

        display(Markdown(table))

    def print(self, metric: str = "f1"):
        """Print the evaluation report.

        Args:
            metric (str, optional): Metric to use for results evaluation. Defaults to "f1".
        """
        print(self.to_markdown(metric))
