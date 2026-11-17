"""Unweighted L2 logistic baseline; no search, class weights or campaign policy."""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from online_retail.modeling.metrics import evaluate
from online_retail.modeling.preprocessing import logistic_preprocessor
from online_retail.modeling.splits import temporal_folds

DERIVED_RFM = {"r_score", "f_score", "m_score", "rfm_score", "rfm_segment"}


def logistic_baseline(columns: list[str]) -> Pipeline:
    """C=1, default L2 penalty; all preprocessing is fitted with the model."""
    return Pipeline(
        [
            ("preprocess", logistic_preprocessor(columns)),
            ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000)),
        ]
    )


def fit_baseline(pipeline, x, y):
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(x, y)
    return pipeline


def run_baseline(data_dir: Path, output_dir: Path) -> dict:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest.get("target") != "RepurchaseWithin50Days":
        raise ValueError("Rebuild processed datasets for the binary target before modeling")
    train = pd.read_parquet(data_dir / "train.parquet").reset_index(drop=True)
    validation = pd.read_parquet(data_dir / "validation.parquet").reset_index(drop=True)
    if train.cutoff.max() + pd.Timedelta(days=50) > validation.cutoff.min():
        raise ValueError("Training outcomes are not mature before validation")
    columns = [c for c in manifest["feature_columns"] if c not in DERIVED_RFM]
    if set(columns) & {"Customer ID", "cutoff", "RepurchaseWithin50Days"}:
        raise ValueError("Keys and target must not be predictors")
    fold_reports = []
    for fold, (fit_rows, valid_rows) in enumerate(temporal_folds(train), 1):
        fit, valid = train.iloc[fit_rows], train.iloc[valid_rows]
        model = fit_baseline(logistic_baseline(columns), fit[columns], fit.RepurchaseWithin50Days)
        dummy = DummyClassifier(strategy="prior").fit(fit[columns], fit.RepurchaseWithin50Days)
        for name, fitted in (("logistic_l2", model), ("prior", dummy)):
            fold_reports.append(
                {
                    "fold": fold,
                    "model": name,
                    "train_log_loss": evaluate(fit, fitted.predict_proba(fit[columns]))["log_loss"],
                    **evaluate(valid, fitted.predict_proba(valid[columns])),
                }
            )
    model = fit_baseline(logistic_baseline(columns), train[columns], train.RepurchaseWithin50Days)
    dummy = DummyClassifier(strategy="prior").fit(train[columns], train.RepurchaseWithin50Days)
    comparison = pd.DataFrame(
        [
            {"model": name, **evaluate(validation, fitted.predict_proba(validation[columns]))}
            for name, fitted in (("prior", dummy), ("logistic_l2", model))
        ]
    )
    predictions = validation[["Customer ID", "cutoff", "RepurchaseWithin50Days"]].copy()
    predictions[["p_class_0", "p_class_1"]] = model.predict_proba(validation[columns])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "logistic_baseline.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    pd.DataFrame(fold_reports).to_csv(output_dir / "cv_metrics.csv", index=False)
    comparison.to_csv(output_dir / "validation_metrics.csv", index=False)
    predictions.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    (output_dir / "run.json").write_text(
        json.dumps(
            {
                "model": "L2 logistic regression",
                "C": 1.0,
                "class_weights": None,
                "feature_columns": columns,
                "hyperparameter_search": False,
                "test_evaluated": False,
                "snapshot_frequency": "weekly",
                "target": "RepurchaseWithin50Days",
                "positive_class": 1,
                "horizon_days": 50,
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "cv_metrics": pd.DataFrame(fold_reports),
        "validation_metrics": comparison,
        "validation_predictions": predictions,
    }


def plot_calibration(predictions: pd.DataFrame):
    """Reliability diagnostic only; no probability recalibration is fitted."""
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for k in (1,):
        observed, predicted = calibration_curve(
            predictions.RepurchaseWithin50Days.eq(k),
            predictions[f"p_class_{k}"],
            n_bins=10,
            strategy="quantile",
        )
        ax.plot(predicted, observed, marker="o", label="Purchase within 50 days")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed class fraction",
        title="External-validation calibration",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    return fig, ax
