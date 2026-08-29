from itertools import combinations
from math import isfinite
from pathlib import Path
from typing import Union

import numpy as np
from IPython.display import Markdown, display
from tabulate import tabulate
from unified_io import read_json

from .datasets import (
    DATASETS,
    UNSAFE_ONLY_DATASETS,
    load_dataset,
    load_dataset_with_fingerprint,
)
from .evaluate import evaluate
from .utils import file_sha256


def _count_and_rate(count: int, denominator: int) -> dict:
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator if denominator else None,
    }


def _normalize_thresholds(models: list[str], thresholds) -> dict[str, float]:
    if isinstance(thresholds, dict):
        if set(thresholds) != set(models):
            raise ValueError("thresholds must name every selected model exactly once")
        result = thresholds
    else:
        result = {model_name: thresholds for model_name in models}

    normalized = {}
    for model_name, threshold in result.items():
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError(f"threshold for {model_name!r} must be numeric")
        if not isfinite(threshold) or threshold < 0 or threshold > 1:
            raise ValueError(f"threshold for {model_name!r} must be in [0, 1]")
        normalized[model_name] = float(threshold)
    return normalized


def _normalize_configurations(
    models: list[str], model_configurations: dict | None
) -> tuple[dict[str, dict | None], str]:
    if model_configurations is None:
        return {model_name: None for model_name in models}, "not_declared"
    if not isinstance(model_configurations, dict) or set(model_configurations) != set(
        models
    ):
        raise ValueError(
            "model_configurations must name every selected model exactly once"
        )
    normalized = {}
    for model_name, configuration in model_configurations.items():
        if configuration is not None and not isinstance(configuration, dict):
            raise TypeError(
                f"model_configuration for {model_name!r} must be an object or None"
            )
        normalized[model_name] = configuration
    return normalized, "caller_declared_at_report"


def _require_same_ids(label: str, expected: set, actual: set) -> None:
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{label} must have exactly the dataset IDs; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _read_json_with_fingerprint(path: Path) -> tuple[dict, str]:
    fingerprint_before_read = file_sha256(path)
    payload = read_json(path)
    if file_sha256(path) != fingerprint_before_read:
        raise RuntimeError(f"Prediction file {path} changed while it was being read")
    return payload, fingerprint_before_read


def _verify_manifest(
    manifest_path: Path,
    *,
    dataset_code_name: str,
    model_name: str,
    item_count: int,
    dataset_sha256: str,
    prediction_sha256: str,
) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing joint-evidence manifest: {manifest_path}. "
            "Re-run the benchmark to create verified input provenance."
        )
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise TypeError(f"Manifest {manifest_path} must be a JSON object")
    required = {
        "schema_version",
        "dataset",
        "model",
        "n_items",
        "dataset_sha256",
        "prediction_sha256",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(
            f"Manifest {manifest_path} is missing required fields: {sorted(missing)}"
        )
    expected = {
        "schema_version": 1,
        "dataset": dataset_code_name,
        "model": model_name,
        "n_items": item_count,
        "dataset_sha256": dataset_sha256,
        "prediction_sha256": prediction_sha256,
    }
    for field, value in expected.items():
        if manifest[field] != value:
            raise ValueError(
                f"Manifest {manifest_path} has {field}={manifest[field]!r}, "
                f"expected {value!r}"
            )
    return manifest


def _joint_evidence(
    y_true: dict,
    predictions: dict[str, dict],
    thresholds: dict[str, float],
    model_configurations: dict[str, dict | None],
    model_configuration_status: str,
    dataset_name: str,
    dataset_sha256: str,
    manifests: dict[str, dict | None],
    provenance_status: str,
) -> dict:
    """Compute a static full-shadow OR report from verified prediction maps."""
    models = list(predictions)
    if len(models) < 2:
        raise ValueError("a joint-evidence report needs at least two models")
    if len(set(models)) != len(models):
        raise ValueError("a joint-evidence report needs unique model names")
    if not y_true:
        raise ValueError("a joint-evidence report needs at least one labelled item")
    if not all(isinstance(label, bool) for label in y_true.values()):
        raise ValueError("ground-truth labels must be bool")

    item_ids = set(y_true)
    decisions = {}
    for model_name, scores in predictions.items():
        _require_same_ids(f"predictions for {model_name!r}", item_ids, set(scores))
        for item_id, score in scores.items():
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError(
                    f"prediction for {model_name!r} item {item_id!r} is not numeric"
                )
            if not isfinite(score) or score < 0 or score > 1:
                raise ValueError(
                    f"prediction for {model_name!r} item {item_id!r} must be a "
                    "finite probability in [0, 1]"
                )
        decisions[model_name] = {
            item_id: score > thresholds[model_name] for item_id, score in scores.items()
        }

    positive_ids = [item_id for item_id, label in y_true.items() if label]
    benign_ids = [item_id for item_id, label in y_true.items() if not label]
    if not positive_ids:
        raise ValueError("static all-miss is undefined without positive items")

    catches = {
        model_name: {
            item_id for item_id in positive_ids if decisions[model_name][item_id]
        }
        for model_name in models
    }
    union = set().union(*catches.values())
    all_miss = set(positive_ids) - union

    residual_coverage = []
    caught_so_far = set()
    for model_name in models:
        new_catches = catches[model_name] - caught_so_far
        prior_misses = len(positive_ids) - len(caught_so_far)
        residual_coverage.append(
            {
                "model": model_name,
                "prior_misses": prior_misses,
                "new_catches": len(new_catches),
                "new_catch_rate_among_prior_misses": (
                    len(new_catches) / prior_misses if prior_misses else None
                ),
            }
        )
        caught_so_far |= catches[model_name]

    benign_flags = {
        model_name: {
            item_id for item_id in benign_ids if decisions[model_name][item_id]
        }
        for model_name in models
    }
    benign_union = set().union(*benign_flags.values()) if benign_ids else set()

    return {
        "schema_version": 1,
        "estimand": "static_full_shadow_or",
        "population": {
            "dataset": dataset_name,
            "dataset_sha256": dataset_sha256,
            "items": len(y_true),
            "positive_label": True,
            "positive_items": len(positive_ids),
            "benign_items": len(benign_ids),
        },
        "exposure": {
            "policy": "error_on_missing_or_extra_ids",
            "all_models_scored_all_items": True,
        },
        "decision_rule": {
            "per_model_thresholds": thresholds,
            "flag": "score > threshold",
            "composition": "parallel full-shadow OR",
            "stack_order": models,
        },
        "model_configurations": {
            "status": model_configuration_status,
            "by_model": model_configurations,
        },
        "input_provenance": {
            "status": provenance_status,
            "manifests": manifests,
        },
        "per_guard_detection": {
            model_name: _count_and_rate(len(catches[model_name]), len(positive_ids))
            for model_name in models
        },
        "union_detection": _count_and_rate(len(union), len(positive_ids)),
        "static_all_miss": _count_and_rate(len(all_miss), len(positive_ids)),
        "ordered_residual_coverage": residual_coverage,
        "pairwise_intersections": [
            {
                "models": [left_model, right_model],
                "both_catch": len(catches[left_model] & catches[right_model]),
                "both_miss": len(
                    set(positive_ids) - catches[left_model] - catches[right_model]
                ),
            }
            for left_model, right_model in combinations(models, 2)
        ],
        "benign_union_flag_rate": (
            _count_and_rate(len(benign_union), len(benign_ids)) if benign_ids else None
        ),
        "uncertainty": {
            "status": "not_reported",
            "reason": (
                "Fixed evaluated population; no sampling design or inferential "
                "target was declared."
            ),
        },
        "non_claims": [
            "Not sequential route risk.",
            "Not adaptive robustness.",
            "Not calibration, causality, or deployment safety.",
        ],
    }


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

    def joint_evidence(
        self,
        thresholds: float | dict[str, float] = 0.5,
        model_configurations: dict | None = None,
        require_manifest: bool = True,
    ) -> dict:
        """Build one static joint-evidence report for each selected dataset.

        The selected model order is the declared parallel-stack order used for
        residual attribution. By default, every saved prediction map needs the
        sidecar manifest written by a post-feature benchmark run. Legacy maps
        can be inspected only with ``require_manifest=False`` and are marked
        as unverified input provenance.
        """
        if len(self.models) < 2:
            raise ValueError("a joint-evidence report needs at least two models")
        if len(set(self.models)) != len(self.models):
            raise ValueError("a joint-evidence report needs unique model names")
        normalized_thresholds = _normalize_thresholds(self.models, thresholds)
        normalized_configurations, configuration_status = _normalize_configurations(
            self.models, model_configurations
        )
        reports = {}
        for dataset_code_name in self.datasets:
            dataset, dataset_sha256 = load_dataset_with_fingerprint(dataset_code_name)
            dataset_name = DATASETS.get(dataset_code_name, dataset.name)
            item_ids = [sample["id"] for sample in dataset.dataset]
            if len(item_ids) != len(set(item_ids)):
                raise ValueError(
                    f"Dataset {dataset_code_name!r} contains duplicate item IDs"
                )
            y_true = dataset.get_ground_truth("test")
            prediction_paths = {
                model_name: self.out_dir / dataset_code_name / f"{model_name}.json"
                for model_name in self.models
            }
            predictions = {}
            prediction_sha256 = {}
            for model_name, path in prediction_paths.items():
                predictions[model_name], prediction_sha256[model_name] = (
                    _read_json_with_fingerprint(path)
                )
            if file_sha256(dataset.path / "test.jsonl") != dataset_sha256:
                raise RuntimeError(
                    f"Dataset {dataset_code_name!r} changed while joint evidence was being prepared"
                )
            for model_name, path in prediction_paths.items():
                if file_sha256(path) != prediction_sha256[model_name]:
                    raise RuntimeError(
                        f"Prediction file {path} changed while joint evidence was being prepared"
                    )
            manifests = {
                model_name: (
                    _verify_manifest(
                        self.out_dir
                        / dataset_code_name
                        / f"{model_name}.manifest.json",
                        dataset_code_name=dataset_code_name,
                        model_name=model_name,
                        item_count=len(y_true),
                        dataset_sha256=dataset_sha256,
                        prediction_sha256=prediction_sha256[model_name],
                    )
                    if require_manifest
                    else None
                )
                for model_name in self.models
            }
            reports[dataset_name] = _joint_evidence(
                y_true,
                predictions,
                dataset_name=dataset_name,
                dataset_sha256=dataset_sha256,
                thresholds=normalized_thresholds,
                model_configurations=normalized_configurations,
                model_configuration_status=configuration_status,
                manifests=manifests,
                provenance_status=(
                    "verified_sidecar_manifest"
                    if require_manifest
                    else "unverified_legacy_maps"
                ),
            )
        return reports

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
