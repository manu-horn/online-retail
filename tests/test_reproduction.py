import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from online_retail.modeling.baseline import fit_baseline
from online_retail.modeling.reproduction import fitting_population, reproduce_fit
from online_retail.serving.bundle import checksum


@pytest.fixture
def reproduction_inputs(tmp_path):
    data, reference = tmp_path / "data", tmp_path / "reference"
    data.mkdir()
    reference.mkdir()
    refit = pd.DataFrame(
        [
            {
                "Customer ID": str(label),
                "cutoff": date,
                "RepurchaseWithin50Days": label,
                "recency_days": 2.0 if label else 60.0,
                "last_gap_days": 4.0 if label else np.nan,
            }
            for date in pd.date_range("2010-03-01", "2010-05-24", freq="W-MON")
            for label in (0, 1)
        ]
    )
    frozen = {
        "model": "xgboost",
        "feature_columns": ["recency_days", "last_gap_days"],
        "params": {"n_estimators": 4, "max_depth": 2},
        "threshold": 0.62,
        "fit_last_cutoff": "2010-03-22",
        "fit_rows": 8,
        "policy_first_cutoff": "2010-05-17",
        "policy_last_cutoff": "2010-05-24",
        "policy_rows": 4,
    }
    fit, policy = fitting_population(refit, frozen)
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", XGBClassifier(**frozen["params"], random_state=42, n_jobs=2)),
        ]
    )
    model = fit_baseline(pipeline, fit[frozen["feature_columns"]], fit.RepurchaseWithin50Days)
    (reference / "model.pkl").write_bytes(pickle.dumps(model))
    (reference / "frozen_policy.json").write_text(json.dumps(frozen))
    predictions = policy[["Customer ID", "cutoff", "RepurchaseWithin50Days"]].copy()
    predictions[["p_class_0", "p_class_1"]] = model.predict_proba(policy[frozen["feature_columns"]])
    predictions.to_parquet(reference / "policy_predictions.parquet", index=False)
    refit.to_parquet(data / "refit.parquet", index=False)
    fit.to_parquet(data / "train.parquet", index=False)
    (data / "manifest.json").write_text(
        json.dumps({"target": "RepurchaseWithin50Days", "horizon_days": 50})
    )
    # Deliberately unreadable: reproduction must never load final test data.
    (data / "test.parquet").write_text("DO NOT READ TEST")
    return data, reference


def test_reproduction_preserves_reference_and_matches_policy_without_test(reproduction_inputs):
    data, reference = reproduction_inputs
    before = {p.name: checksum(p) for p in reference.iterdir()}
    output = data.parent / "reproduced"
    report = reproduce_fit(data, reference, output)
    assert report["verification"]["max_probability_difference"] == 0
    assert report["verification"]["decisions_match"]
    assert not report["test_evaluated"] and not report["hyperparameter_search"]
    assert report["fit_rows"] == 8
    assert before == {p.name: checksum(p) for p in reference.iterdir()}
    assert (output / "frozen_policy.json").read_bytes() == (
        reference / "frozen_policy.json"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        reproduce_fit(data, reference, output)


def test_reproduction_rejects_changed_population(reproduction_inputs):
    data, reference = reproduction_inputs
    refit = pd.read_parquet(data / "refit.parquet").iloc[1:]
    refit.to_parquet(data / "refit.parquet", index=False)
    with pytest.raises(ValueError, match="population"):
        reproduce_fit(data, reference, data.parent / "invalid")
    assert not (data.parent / "invalid").exists()


def test_failed_parity_publishes_no_model(reproduction_inputs):
    data, reference = reproduction_inputs
    path = reference / "policy_predictions.parquet"
    predictions = pd.read_parquet(path)
    predictions["p_class_1"] = 0.99
    predictions.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="parity failed"):
        reproduce_fit(data, reference, data.parent / "invalid")
    assert not (data.parent / "invalid").exists()


def test_reproduction_rejects_label_overlap(reproduction_inputs):
    data, reference = reproduction_inputs
    frozen = json.loads((reference / "frozen_policy.json").read_text())
    frozen["policy_first_cutoff"] = "2010-03-29"
    with pytest.raises(ValueError, match="mature"):
        fitting_population(pd.read_parquet(data / "refit.parquet"), frozen)


def test_reproduction_is_tracked_as_fitting_not_import(reproduction_inputs):
    from mlflow.tracking import MlflowClient

    from online_retail.modeling.tracking import TrackingConfig
    from online_retail.modeling.workflows import run_reproduction

    data, reference = reproduction_inputs
    root = data.parent
    config = TrackingConfig(root, f"sqlite:///{root / 'mlflow.db'}", "reproduction-test")
    output = root / "tracked-reproduction"
    run_reproduction(data, reference, output, config)
    run_id = json.loads((output / "mlflow_run.json").read_text())["run_id"]
    run = MlflowClient(tracking_uri=config.uri).get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["workflow.mode"] == "fixed_parameter_reproduction"
    assert run.data.tags["reproduction.test_data_loaded"] == "false"
    assert run.data.metrics["reported.verification.max_probability_difference"] == 0
    assert run.data.tags["model.uri"]
