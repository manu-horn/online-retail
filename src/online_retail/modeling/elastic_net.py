"""Small chronological elastic-net search, independent of campaign thresholds."""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from online_retail.modeling.baseline import DERIVED_RFM, fit_baseline, logistic_baseline
from online_retail.modeling.metrics import evaluate
from online_retail.modeling.splits import temporal_folds


def run_elastic_net(data_dir: Path, output_dir: Path) -> dict:
    """Select regularization on training folds; evaluate later validation once."""
    manifest = json.loads((data_dir / "manifest.json").read_text())
    train = pd.read_parquet(data_dir / "train.parquet")
    validation = pd.read_parquet(data_dir / "validation.parquet")
    columns = [c for c in manifest["feature_columns"] if c not in DERIVED_RFM]
    reports = []

    def pipeline(c, ratio):
        model = logistic_baseline(columns)
        model.set_params(
            model=LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                C=c,
                l1_ratio=ratio,
                max_iter=10000,
                random_state=42,
            )
        )
        return model

    for c in (0.01, 0.1, 1.0):
        for ratio in (0.25, 0.75):
            for fold, (fit_rows, valid_rows) in enumerate(temporal_folds(train), 1):
                fit, valid = train.iloc[fit_rows], train.iloc[valid_rows]
                model = fit_baseline(pipeline(c, ratio), fit[columns], fit.RepurchaseWithin50Days)
                reports.append(
                    {
                        "C": c,
                        "l1_ratio": ratio,
                        "fold": fold,
                        **evaluate(valid, model.predict_proba(valid[columns])),
                    }
                )
    cv = pd.DataFrame(reports)
    ranking = cv.groupby(["C", "l1_ratio"], as_index=False).mean(numeric_only=True)
    ranking = ranking.sort_values("weekly_log_loss").reset_index(drop=True)
    best = ranking.iloc[0]
    model = fit_baseline(
        pipeline(float(best.C), float(best.l1_ratio)), train[columns], train.RepurchaseWithin50Days
    )
    predictions = validation[["Customer ID", "cutoff", "RepurchaseWithin50Days"]].copy()
    predictions[["p_class_0", "p_class_1"]] = model.predict_proba(validation[columns])
    metrics = pd.DataFrame(
        [
            {
                "model": "logistic_elastic_net",
                **evaluate(validation, predictions[["p_class_0", "p_class_1"]].to_numpy()),
            }
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cv.to_csv(output_dir / "cv_metrics.csv", index=False)
    ranking.to_csv(output_dir / "candidate_metrics.csv", index=False)
    metrics.to_csv(output_dir / "validation_metrics.csv", index=False)
    predictions.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    (output_dir / "run.json").write_text(
        json.dumps(
            {
                "model": "logistic_elastic_net",
                "feature_columns": columns,
                "best_params": {"C": float(best.C), "l1_ratio": float(best.l1_ratio)},
                "objective_metric": "equal_fold_mean_weekly_log_loss",
                "random_state": 42,
                "test_evaluated": False,
            },
            indent=2,
        )
    )
    return {
        "candidate_metrics": ranking,
        "validation_metrics": metrics,
        "validation_predictions": predictions,
    }
