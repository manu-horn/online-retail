import json
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from online_retail.modeling.tracking import TrackingConfig
from online_retail.modeling.workflows import import_results, run_training


@pytest.fixture
def tracked_data(tmp_path):
    root = tmp_path / "project"
    data = root / "data"
    data.mkdir(parents=True)
    rows = [
        {
            "Customer ID": f"customer-{label}",
            "cutoff": cutoff,
            "RepurchaseWithin50Days": label,
            "recency_days": 1.0 + 40 * (1 - label),
            "invoice_count": 1 + label,
            "last_gap_days": float("nan") if label == 0 else 5.0,
        }
        for cutoff in pd.date_range("2010-03-01", "2010-10-11", freq="W-MON")
        for label in [0, 1]
    ]
    train = pd.DataFrame(rows)
    validation = train.iloc[:4].copy()
    validation["cutoff"] = pd.Timestamp("2010-12-06")
    train.to_parquet(data / "train.parquet", index=False)
    validation.to_parquet(data / "validation.parquet", index=False)
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "target": "RepurchaseWithin50Days",
                "feature_columns": ["recency_days", "invoice_count", "last_gap_days"],
            }
        )
    )
    config = TrackingConfig(root, f"sqlite:///{root / 'tracking.db'}", "integration-test")
    return root, data, config


def test_tracked_training_logs_reproducible_probability_model(tracked_data):
    root, data, config = tracked_data
    output = root / "output"
    result = run_training("baseline", data, output, config)
    assert len(result["validation_metrics"]) == 2
    run_id = json.loads((output / "mlflow_run.json").read_text())["run_id"]
    client = MlflowClient(tracking_uri=config.uri)
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["workflow.mode"] == "training"
    assert "validation.logistic_l2.roc_auc" in run.data.metrics
    model = mlflow.pyfunc.load_model(run.data.tags["model.uri"])
    x = pd.read_parquet(data / "validation.parquet")[
        ["recency_days", "invoice_count", "last_gap_days"]
    ]
    probabilities = np.asarray(model.predict(x))
    with (output / "logistic_baseline.pkl").open("rb") as handle:
        original = pickle.load(handle)
    assert np.allclose(probabilities, original.predict_proba(x))
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert set(run.data.tags) >= {"source.model_sha256", "git.commit"}
    assert (
        Path(client.download_artifacts(run_id, "provenance")) / "dataset_checksums.json"
    ).exists()
    assert "validation_predictions.parquet" not in {
        Path(a.path).name for a in client.list_artifacts(run_id, "reports")
    }
    # Import reuses the same artifact without modifying it or fitting again.
    before = (output / "logistic_baseline.pkl").read_bytes()
    imported = import_results(output, data, config)
    assert client.get_run(imported).data.tags["workflow.mode"] == "historical_artifact_import"
    assert (output / "logistic_baseline.pkl").read_bytes() == before


def test_failed_training_is_marked_failed(tracked_data):
    root, data, config = tracked_data
    with pytest.raises(ValueError, match="Unknown training"):
        run_training("invalid", data, root / "failed", config)
    client = MlflowClient(tracking_uri=config.uri)
    experiment = client.get_experiment_by_name(config.experiment)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1 and runs[0].info.status == "FAILED"
    assert mlflow.active_run() is None
