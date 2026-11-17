"""Optuna tree comparisons with mature-label expanding temporal validation."""

import json
import pickle
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from online_retail.modeling.baseline import DERIVED_RFM, fit_baseline
from online_retail.modeling.metrics import evaluate
from online_retail.modeling.splits import temporal_folds


def tree_pipeline(name, trial):
    """Original feature scales; median imputation fits within each fold."""
    common = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "random_state": 42,
        "n_jobs": 2,
    }
    if name == "random_forest":
        model = RandomForestClassifier(
            **common,
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 150),
            max_features=trial.suggest_float("max_features", 0.4, 1.0),
            max_samples=trial.suggest_float("max_samples", 0.6, 1.0),
        )
    else:
        common.update(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 30, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 0.001, 10, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        )
        if name == "lightgbm":
            model = LGBMClassifier(
                **common,
                verbosity=-1,
                num_leaves=trial.suggest_int("num_leaves", 7, 31),
                min_child_samples=trial.suggest_int("min_child_samples", 30, 150),
            )
        elif name == "xgboost":
            model = XGBClassifier(
                **common,
                tree_method="hist",
                eval_metric="logloss",
                min_child_weight=trial.suggest_float("min_child_weight", 5, 50),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
            )
        else:
            raise ValueError(f"Unknown model: {name}")
    return Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("model", model),
        ]
    )


def run_tree_models(
    data_dir: Path, output_dir: Path, n_trials=15, objective_metric="weekly_log_loss"
):
    """Tune on train only, then report external validation; test never loaded."""
    if objective_metric not in {"weekly_log_loss", "weekly_roc_auc"}:
        raise ValueError("Unsupported optimization metric")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest["target"] != "RepurchaseWithin50Days":
        raise ValueError("Expected binary datasets")
    train = pd.read_parquet(data_dir / "train.parquet").reset_index(drop=True)
    valid = pd.read_parquet(data_dir / "validation.parquet").reset_index(drop=True)
    if train.cutoff.max() + pd.Timedelta(days=50) > valid.cutoff.min():
        raise ValueError("Training labels must mature before validation")
    columns = [c for c in manifest["feature_columns"] if c not in DERIVED_RFM]
    folds = list(temporal_folds(train))
    results = []
    for name in ("lightgbm", "xgboost", "random_forest"):
        destination = output_dir / name
        destination.mkdir(parents=True, exist_ok=True)
        reports = []

        def objective(trial, name=name, reports=reports):
            losses, weeks = [], []
            for fold, (fit_rows, val_rows) in enumerate(folds, 1):
                fit, val = train.iloc[fit_rows], train.iloc[val_rows]
                model = fit_baseline(
                    tree_pipeline(name, trial), fit[columns], fit.RepurchaseWithin50Days
                )
                metrics = evaluate(val, model.predict_proba(val[columns]))
                reports.append(
                    {
                        "trial": trial.number,
                        "fold": fold,
                        "train_log_loss": evaluate(fit, model.predict_proba(fit[columns]))[
                            "log_loss"
                        ],
                        **metrics,
                    }
                )
                probabilities = model.predict_proba(val[columns])[:, 1]
                weekly_auc = [
                    roc_auc_score(
                        group.RepurchaseWithin50Days,
                        pd.Series(probabilities, index=val.index).loc[group.index],
                    )
                    for _, group in val.groupby("cutoff")
                ]
                reports[-1]["weekly_roc_auc"] = float(np.mean(weekly_auc))
                losses.append(reports[-1][objective_metric])
                weeks.append(val.cutoff.nunique())
            return float(np.average(losses, weights=weeks))

        study = optuna.create_study(
            direction="maximize" if objective_metric == "weekly_roc_auc" else "minimize",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
            pruner=optuna.pruners.NopPruner(),
        )
        study.optimize(objective, n_trials=n_trials)
        model = fit_baseline(
            tree_pipeline(name, optuna.trial.FixedTrial(study.best_params)),
            train[columns],
            train.RepurchaseWithin50Days,
        )
        probabilities = model.predict_proba(valid[columns])
        results.append({"model": name, **evaluate(valid, probabilities)})
        predictions = valid[["Customer ID", "cutoff", "RepurchaseWithin50Days"]].copy()
        predictions[["p_class_0", "p_class_1"]] = probabilities
        predictions.to_parquet(destination / "validation_predictions.parquet", index=False)
        pd.DataFrame(reports).to_csv(destination / "cv_metrics.csv", index=False)
        study.trials_dataframe().to_csv(destination / "trials.csv", index=False)
        pd.DataFrame([results[-1]]).to_csv(destination / "validation_metrics.csv", index=False)
        (destination / "run.json").write_text(
            json.dumps(
                {
                    "best_params": study.best_params,
                    "best_cv_value": study.best_value,
                    "objective_metric": objective_metric,
                    "n_trials": n_trials,
                    "feature_columns": columns,
                    "test_evaluated": False,
                    "seed": 42,
                    "objective": objective_metric,
                },
                indent=2,
            )
        )
        with (destination / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)
    return pd.DataFrame(results)
