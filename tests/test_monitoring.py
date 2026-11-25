import json

import numpy as np
import pandas as pd
import pytest
from container_smoke import synthetic_inputs

from online_retail.features.snapshots import customer_features
from online_retail.logging import configure_logging
from online_retail.monitoring.core import build_reference, history_fingerprint, monitor_batch
from online_retail.monitoring.outcomes import evaluate_outcomes
from online_retail.serving.bundle import load_bundle
from online_retail.serving.inference import predict_features
from online_retail.serving.workflow import run_scoring


@pytest.fixture
def monitored(tmp_path):
    inputs = tmp_path / "inputs"
    synthetic_inputs(inputs)
    purchases = pd.read_parquet(inputs / "purchases.parquet")
    historical = customer_features(purchases, "2011-04-11")
    path = tmp_path / "reference-data.parquet"
    historical.to_parquet(path, index=False)  # No target column exists.
    build_reference(path, inputs / "bundle", tmp_path / "reference", {"minimum_rows": 1})
    return inputs, tmp_path / "reference/reference.json", historical


def test_reference_uses_one_policy_population_without_labels(monitored, tmp_path):
    inputs, reference_path, historical = monitored
    reference = json.loads(reference_path.read_text())
    assert reference["rows"] == 1 and reference["cutoff"].startswith("2011-04-11")
    assert reference["performance_reference"] is False
    model, policy, manifest = load_bundle(inputs / "bundle")
    scores = predict_features(model, policy, historical)
    same = monitor_batch(historical, scores, reference_path, manifest, policy["threshold"])
    assert same["status"] == "ok"
    assert same["probability_distribution_tv"] == 0
    assert all(v["distribution_tv"] in (0, None) for v in same["features"].values())
    with pytest.raises(FileExistsError):
        build_reference(
            tmp_path / "reference-data.parquet", inputs / "bundle", reference_path.parent
        )


def test_drift_uses_fixed_bins_and_detects_constant_to_new_values(monitored):
    inputs, reference_path, historical = monitored
    model, policy, manifest = load_bundle(inputs / "bundle")
    changed = historical.copy()
    changed["monetary_total"] *= 100000
    changed["last_gap_days"] = np.nan
    result = monitor_batch(
        changed,
        predict_features(model, policy, changed),
        reference_path,
        manifest,
        policy["threshold"],
    )
    assert result["features"]["monetary_total"]["distribution_tv"] == 1
    assert "feature_distribution:monetary_total" in result["warnings"]
    assert "missingness:last_gap_days" in result["warnings"]
    assert not result["labels_used"] and not result["performance_degradation_established"]


def test_monitored_score_is_same_prediction_and_publishes_report(monitored, tmp_path):
    inputs, reference, _ = monitored
    original = run_scoring(
        inputs / "purchases.parquet",
        inputs / "bundle",
        "2011-06-06",
        "2011-06-06",
        tmp_path / "plain",
    )
    output = tmp_path / "monitored"
    scored = run_scoring(
        inputs / "purchases.parquet",
        inputs / "bundle",
        "2011-06-06",
        "2011-06-06",
        output,
        monitoring_reference=reference,
    )
    np.testing.assert_array_equal(scored.purchase_probability, original.purchase_probability)
    assert scored.contact.tolist() == original.contact.tolist()
    assert scored.history_group.tolist() == ["4+", "1"]
    assert set(scored["Customer ID"]) == {"A", "B"}
    assert all(
        (output / name).exists() for name in ("monitoring.json", "monitoring.md", "monitoring.png")
    )
    assert "RepurchaseWithin50Days" not in scored
    report = json.loads((output / "monitoring.json").read_text())
    assert report["rows"] == 2 and report["status"] == "warning"


def test_bad_reference_logs_failure_without_publishing(monitored, tmp_path):
    inputs, reference, _ = monitored
    value = json.loads(reference.read_text())
    value["model_version"] = "different"
    reference.write_text(json.dumps(value))
    log = tmp_path / "events.jsonl"
    logger = configure_logging(log)
    with pytest.raises(ValueError, match="different model"):
        run_scoring(
            inputs / "purchases.parquet",
            inputs / "bundle",
            "2011-06-06",
            "2011-06-06",
            tmp_path / "bad",
            logger,
            monitoring_reference=reference,
        )
    assert not (tmp_path / "bad").exists()
    assert json.loads(log.read_text().splitlines()[-1])["event"] == "scoring_failed"


def score_fixture(monitored, tmp_path):
    inputs, reference, _ = monitored
    output = tmp_path / "scored"
    run_scoring(
        inputs / "purchases.parquet",
        inputs / "bundle",
        "2011-06-06",
        "2011-06-06",
        output,
        monitoring_reference=reference,
    )
    return inputs, output


def test_mature_followup_is_exclusive_and_only_evaluates_scored_customers(monitored, tmp_path):
    inputs, run = score_fixture(monitored, tmp_path)
    path = inputs / "purchases.parquet"
    purchases = pd.read_parquet(path)
    additions = purchases.iloc[:2].copy()
    additions["Customer ID"] = ["B", "A"]
    additions["Invoice"] = ["new-B", "late-A"]
    additions["InvoiceDate"] = pd.to_datetime(["2011-06-06", "2011-07-26"])
    pd.concat([purchases, additions], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="50-day"):
        evaluate_outcomes(run, path, "2011-07-25", tmp_path / "too-early")
    assert not (tmp_path / "too-early").exists()
    report = evaluate_outcomes(run, path, "2011-07-26", tmp_path / "outcomes")
    assert report["overall"]["snapshots"] == 2 and report["overall"]["buyers"] == 1
    assert report["history_groups"]["1"]["buyers"] == 1
    assert report["history_groups"]["4+"]["buyers"] == 0
    assert report["history_groups"]["4+"]["roc_auc"] is None
    assert report["history_unchanged"]
    with pytest.raises(FileExistsError):
        evaluate_outcomes(run, path, "2011-07-26", tmp_path / "outcomes")


def test_changed_history_is_detected_but_reordering_is_allowed(monitored, tmp_path):
    inputs, run = score_fixture(monitored, tmp_path)
    path = inputs / "purchases.parquet"
    purchases = pd.read_parquet(path)
    assert history_fingerprint(purchases, "2011-06-06") == history_fingerprint(
        purchases.iloc[::-1], "2011-06-06"
    )
    purchases.iloc[::-1].to_parquet(path, index=False)
    result = evaluate_outcomes(run, path, "2011-07-26", tmp_path / "zero-buyers")
    assert result["overall"]["average_precision"] is None
    assert result["overall"]["roc_auc"] is None
    purchases.loc[0, "Price"] = 100
    purchases.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="history changed"):
        evaluate_outcomes(run, path, "2011-07-26", tmp_path / "changed")
    assert not (tmp_path / "changed").exists()
