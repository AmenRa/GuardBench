import json

import pytest
from unified_io import write_jsonl

from guardbench import Report, benchmark
from guardbench import report as report_module
from guardbench.datasets.custom_dataset import CustomDataset
from guardbench.report import _joint_evidence
from guardbench.utils import file_sha256


def _core_report(labels, predictions):
    models = list(predictions)
    return _joint_evidence(
        labels,
        predictions,
        thresholds={model_name: 0.5 for model_name in models},
        model_configurations={model_name: None for model_name in models},
        model_configuration_status="fixture",
        dataset_name="fixture",
        dataset_sha256="fixture-dataset-sha",
        manifests={model_name: None for model_name in models},
        provenance_status="fixture",
    )


def test_joint_evidence_counts_and_strict_threshold():
    labels = {"0": True, "1": True, "2": True, "3": True, "4": False}
    predictions = {
        "guard_a": {"0": 0.9, "1": 0.8, "2": 0.2, "3": 0.5, "4": 0.9},
        "guard_b": {"0": 0.9, "1": 0.2, "2": 0.9, "3": 0.1, "4": 0.1},
    }

    report = _core_report(labels, predictions)

    assert report["union_detection"] == {
        "count": 3,
        "denominator": 4,
        "rate": 0.75,
    }
    assert report["static_all_miss"] == {
        "count": 1,
        "denominator": 4,
        "rate": 0.25,
    }
    assert (
        report["union_detection"]["count"] + report["static_all_miss"]["count"]
        == report["population"]["positive_items"]
    )
    assert report["benign_union_flag_rate"] == {
        "count": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    assert report["pairwise_intersections"] == [
        {"models": ["guard_a", "guard_b"], "both_catch": 1, "both_miss": 1}
    ]


def test_joint_evidence_refuses_partial_ids_and_empty_positive_population():
    labels = {"0": True, "1": True}
    predictions = {
        "guard_a": {"0": 0.9, "1": 0.1},
        "guard_b": {"0": 0.9},
    }

    with pytest.raises(ValueError, match="exactly the dataset IDs"):
        _core_report(labels, predictions)

    with pytest.raises(ValueError, match="exactly the dataset IDs"):
        _core_report(
            {"0": True},
            {
                "guard_a": {"0": 0.9},
                "guard_b": {"0": 0.9, "unexpected": 0.1},
            },
        )

    with pytest.raises(ValueError, match="without positive items"):
        _core_report(
            {"0": False},
            {"guard_a": {"0": 0.1}, "guard_b": {"0": 0.1}},
        )


def test_joint_evidence_refuses_a_single_guard_and_marks_no_benign_rate_null():
    with pytest.raises(ValueError, match="at least two models"):
        _core_report({"0": True}, {"guard_a": {"0": 0.9}})

    report = _core_report(
        {"0": True, "1": True},
        {"guard_a": {"0": 0.9, "1": 0.1}, "guard_b": {"0": 0.1, "1": 0.9}},
    )
    assert report["benign_union_flag_rate"] is None


def test_benchmark_refuses_a_batch_score_count_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARDBENCH_BASE_PATH", str(tmp_path / "guardbench-data"))
    dataset_name = "score_count_fixture"
    source = tmp_path / "test.jsonl"
    write_jsonl(
        [
            {
                "id": str(index),
                "label": True,
                "conversation": [{"role": "user", "content": str(index)}],
            }
            for index in range(2)
        ],
        source,
    )
    CustomDataset(dataset_name).load_data(source)

    calls = 0

    def return_extra_first_batch(conversations):
        nonlocal calls
        calls += 1
        return [0.9, 0.1] if calls == 1 else [0.9]

    with pytest.raises(ValueError, match="returned 2 score\\(s\\) for a batch of 1"):
        benchmark(
            return_extra_first_batch,
            model_name="short_guard",
            datasets=[dataset_name],
            batch_size=1,
            out_dir=tmp_path / "results",
        )


def test_benchmark_preserves_model_configuration_for_existing_moderators(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GUARDBENCH_BASE_PATH", str(tmp_path / "guardbench-data"))
    dataset_name = "forwarded_configuration_fixture"
    source = tmp_path / "test.jsonl"
    write_jsonl(
        [
            {
                "id": "0",
                "label": True,
                "conversation": [{"role": "user", "content": "sample"}],
            }
        ],
        source,
    )
    CustomDataset(dataset_name).load_data(source)
    received = {}

    def moderate(conversations, model_configuration):
        received["model_configuration"] = model_configuration
        return [0.9 for _ in conversations]

    benchmark(
        moderate,
        model_name="compatible_guard",
        datasets=[dataset_name],
        out_dir=tmp_path / "results",
        model_configuration={"legacy": "still forwarded"},
    )
    assert received["model_configuration"] == {"legacy": "still forwarded"}


def test_benchmark_fingerprints_dataset_before_scoring(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARDBENCH_BASE_PATH", str(tmp_path / "guardbench-data"))
    dataset_name = "pre_score_fingerprint_fixture"
    source = tmp_path / "test.jsonl"
    samples = [
        {
            "id": str(index),
            "label": True,
            "conversation": [{"role": "user", "content": str(index)}],
        }
        for index in range(2)
    ]
    write_jsonl(samples, source)
    dataset = CustomDataset(dataset_name).load_data(source)
    dataset_path = dataset.path / "test.jsonl"
    before_scoring = file_sha256(dataset_path)

    def mutate_source_while_scoring(conversations):
        changed = [dict(sample) for sample in samples]
        changed[0]["conversation"] = [{"role": "user", "content": "changed"}]
        write_jsonl(changed, dataset_path)
        return [0.9 for _ in conversations]

    out_dir = tmp_path / "results"
    with pytest.raises(RuntimeError, match="changed while scoring"):
        benchmark(
            mutate_source_while_scoring,
            model_name="mutating_guard",
            datasets=[dataset_name],
            batch_size=2,
            out_dir=out_dir,
        )

    assert file_sha256(dataset_path) != before_scoring
    assert not (out_dir / dataset_name / "mutating_guard.manifest.json").exists()


def test_report_joint_evidence_verifies_sidecars_and_detects_tampering(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GUARDBENCH_BASE_PATH", str(tmp_path / "guardbench-data"))
    dataset_name = "joint_evidence_fixture"
    source = tmp_path / "test.jsonl"
    samples = [
        {
            "id": str(index),
            "label": index < 4,
            "conversation": [{"role": "user", "content": f"sample {index}"}],
        }
        for index in range(5)
    ]
    write_jsonl(samples, source)
    CustomDataset(dataset_name).load_data(source)
    out_dir = tmp_path / "results"

    benchmark(
        lambda conversations: [0.9, 0.8, 0.2, 0.1, 0.9],
        model_name="guard_a",
        out_dir=out_dir,
        batch_size=5,
        datasets=[dataset_name],
    )
    benchmark(
        lambda conversations: [0.9, 0.2, 0.9, 0.1, 0.1],
        model_name="guard_b",
        out_dir=out_dir,
        batch_size=5,
        datasets=[dataset_name],
    )

    prediction_path = out_dir / dataset_name / "guard_a.json"
    manifest_path = out_dir / dataset_name / "guard_a.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["dataset_sha256"] == file_sha256(
        CustomDataset(dataset_name).path / "test.jsonl"
    )
    assert manifest["prediction_sha256"] == file_sha256(prediction_path)
    assert "model_configuration" not in manifest

    report = Report(
        models=["guard_a", "guard_b"], datasets=[dataset_name], out_dir=out_dir
    )
    undecorated = report.joint_evidence()[dataset_name]
    assert undecorated["model_configurations"] == {
        "status": "not_declared",
        "by_model": {"guard_a": None, "guard_b": None},
    }
    output = report.joint_evidence(
        model_configurations={
            "guard_a": {"model_id": "fixture/a", "revision": "fixture-a"},
            "guard_b": {"model_id": "fixture/b", "revision": "fixture-b"},
        }
    )[dataset_name]
    assert output["input_provenance"]["status"] == "verified_sidecar_manifest"
    assert output["union_detection"]["count"] == 3
    assert output["static_all_miss"]["count"] == 1
    assert output["model_configurations"] == {
        "status": "caller_declared_at_report",
        "by_model": {
            "guard_a": {"model_id": "fixture/a", "revision": "fixture-a"},
            "guard_b": {"model_id": "fixture/b", "revision": "fixture-b"},
        },
    }

    prediction_path.write_text(
        json.dumps({"0": 0.1, "1": 0.1, "2": 0.1, "3": 0.1, "4": 0.1})
    )
    with pytest.raises(ValueError, match="prediction_sha256"):
        report.joint_evidence()


def test_report_joint_evidence_refuses_legacy_maps_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARDBENCH_BASE_PATH", str(tmp_path / "guardbench-data"))
    dataset_name = "legacy_joint_fixture"
    source = tmp_path / "test.jsonl"
    write_jsonl(
        [
            {
                "id": str(index),
                "label": True,
                "conversation": [{"role": "user", "content": str(index)}],
            }
            for index in range(2)
        ],
        source,
    )
    CustomDataset(dataset_name).load_data(source)
    out_dir = tmp_path / "results"
    result_dir = out_dir / dataset_name
    result_dir.mkdir(parents=True)
    (result_dir / "guard_a.json").write_text(json.dumps({"0": 0.9, "1": 0.1}))
    (result_dir / "guard_b.json").write_text(json.dumps({"0": 0.1, "1": 0.9}))

    report = Report(
        models=["guard_a", "guard_b"], datasets=[dataset_name], out_dir=out_dir
    )
    with pytest.raises(FileNotFoundError, match="Missing joint-evidence manifest"):
        report.joint_evidence()

    output = report.joint_evidence(require_manifest=False)[dataset_name]
    assert output["input_provenance"]["status"] == "unverified_legacy_maps"
    assert output["model_configurations"] == {
        "status": "not_declared",
        "by_model": {"guard_a": None, "guard_b": None},
    }


def test_report_joint_evidence_refuses_duplicate_dataset_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARDBENCH_BASE_PATH", str(tmp_path / "guardbench-data"))
    dataset_name = "duplicate_id_fixture"
    source = tmp_path / "test.jsonl"
    write_jsonl(
        [
            {
                "id": "duplicate",
                "label": True,
                "conversation": [{"role": "user", "content": "first"}],
            },
            {
                "id": "duplicate",
                "label": True,
                "conversation": [{"role": "user", "content": "second"}],
            },
        ],
        source,
    )
    CustomDataset(dataset_name).load_data(source)
    out_dir = tmp_path / "results"
    result_dir = out_dir / dataset_name
    result_dir.mkdir(parents=True)
    (result_dir / "guard_a.json").write_text(json.dumps({"duplicate": 0.9}))
    (result_dir / "guard_b.json").write_text(json.dumps({"duplicate": 0.1}))

    report = Report(
        models=["guard_a", "guard_b"], datasets=[dataset_name], out_dir=out_dir
    )
    with pytest.raises(ValueError, match="duplicate item IDs"):
        report.joint_evidence(require_manifest=False)


def test_report_refuses_a_prediction_file_that_changes_while_reading(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GUARDBENCH_BASE_PATH", str(tmp_path / "guardbench-data"))
    dataset_name = "mutating_prediction_fixture"
    source = tmp_path / "test.jsonl"
    write_jsonl(
        [
            {
                "id": str(index),
                "label": True,
                "conversation": [{"role": "user", "content": str(index)}],
            }
            for index in range(2)
        ],
        source,
    )
    CustomDataset(dataset_name).load_data(source)
    out_dir = tmp_path / "results"
    result_dir = out_dir / dataset_name
    result_dir.mkdir(parents=True)
    prediction_path = result_dir / "guard_a.json"
    prediction_path.write_text(json.dumps({"0": 0.9, "1": 0.1}))
    (result_dir / "guard_b.json").write_text(json.dumps({"0": 0.1, "1": 0.9}))
    report = Report(
        models=["guard_a", "guard_b"], datasets=[dataset_name], out_dir=out_dir
    )
    original_read_json = report_module.read_json

    def mutate_after_read(path):
        payload = original_read_json(path)
        if path == prediction_path:
            prediction_path.write_text(json.dumps({"0": 0.1, "1": 0.1}))
        return payload

    monkeypatch.setattr(report_module, "read_json", mutate_after_read)
    with pytest.raises(RuntimeError, match="changed while it was being read"):
        report.joint_evidence(require_manifest=False)


def test_report_refuses_a_dataset_that_changes_while_preparing(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARDBENCH_BASE_PATH", str(tmp_path / "guardbench-data"))
    dataset_name = "mutating_dataset_fixture"
    source = tmp_path / "test.jsonl"
    samples = [
        {
            "id": str(index),
            "label": True,
            "conversation": [{"role": "user", "content": str(index)}],
        }
        for index in range(2)
    ]
    write_jsonl(samples, source)
    dataset = CustomDataset(dataset_name).load_data(source)
    out_dir = tmp_path / "results"
    result_dir = out_dir / dataset_name
    result_dir.mkdir(parents=True)
    (result_dir / "guard_a.json").write_text(json.dumps({"0": 0.9, "1": 0.1}))
    (result_dir / "guard_b.json").write_text(json.dumps({"0": 0.1, "1": 0.9}))
    report = Report(
        models=["guard_a", "guard_b"], datasets=[dataset_name], out_dir=out_dir
    )
    original_load = report_module.load_dataset_with_fingerprint

    def mutate_after_load(name):
        loaded, fingerprint = original_load(name)
        changed = [dict(sample) for sample in samples]
        changed[0]["conversation"] = [{"role": "user", "content": "changed"}]
        write_jsonl(changed, dataset.path / "test.jsonl")
        return loaded, fingerprint

    monkeypatch.setattr(
        report_module, "load_dataset_with_fingerprint", mutate_after_load
    )
    with pytest.raises(
        RuntimeError, match="changed while joint evidence was being prepared"
    ):
        report.joint_evidence(require_manifest=False)
