"""Small chronological reproduction fixture shared by pytest and container checks."""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from online_retail.modeling.baseline import fit_baseline
from online_retail.modeling.reproduction import fitting_population


def make_reproduction_inputs(tmp_path):
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
