"""Frozen ROC-AUC selection, chronological policy holdout and final test."""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from online_retail.modeling.baseline import fit_baseline
from online_retail.modeling.metrics import evaluate
from online_retail.modeling.policy import campaign_cost
from online_retail.modeling.trees import tree_pipeline

COSTS = [[0, 15], [25, 10]]


def policy_partition(refit):
    """Reserve last six scoring weeks; fitting labels must mature before them."""
    weeks = sorted(refit.cutoff.unique())
    if len(weeks) < 7:
        raise ValueError("Need at least seven scoring weeks")
    start = pd.Timestamp(weeks[-6])
    fit = refit.loc[refit.cutoff + pd.Timedelta(days=50) <= start].copy()
    policy = refit.loc[refit.cutoff >= start].copy()
    if fit.empty or policy.empty:
        raise ValueError("Insufficient mature fitting or policy data")
    return fit, policy


def cost_curve(predictions, costs=COSTS):
    """Fixed 0.01 grid plus explicit never-contact policy (handles probability 1)."""
    y, p = predictions.RepurchaseWithin50Days, predictions.p_class_1
    rows = [
        {"threshold": float(t), **campaign_cost(y, p, float(t), costs)}
        for t in np.linspace(0, 1, 101)
    ]
    rows.append(
        {
            "threshold": float("inf"),
            "total_cost": float(np.array(costs)[y, 0].sum()),
            "mean_cost": float(np.array(costs)[y, 0].mean()),
            "contacts": 0,
        }
    )
    return pd.DataFrame(rows)


def run_final(data_dir: Path, search_dir: Path, output: Path):
    """Select solely by internal CV; freeze policy before opening test."""
    if (output / "report.json").exists():
        raise FileExistsError("Final evaluation already exists; load its report rather than retune")
    runs = {
        name: json.loads((search_dir / name / "run.json").read_text())
        for name in ("lightgbm", "xgboost", "random_forest")
    }
    if any(r.get("objective_metric") != "weekly_roc_auc" for r in runs.values()):
        raise ValueError("Expected ROC-AUC searches")
    name = max(runs, key=lambda k: runs[k]["best_cv_value"])
    run = runs[name]
    columns = run["feature_columns"]
    refit = pd.read_parquet(data_dir / "refit.parquet")
    fit, policy = policy_partition(refit)
    model = fit_baseline(
        tree_pipeline(name, optuna.trial.FixedTrial(run["best_params"])),
        fit[columns],
        fit.RepurchaseWithin50Days,
    )
    predictions = policy[["Customer ID", "cutoff", "RepurchaseWithin50Days"]].copy()
    predictions[["p_class_0", "p_class_1"]] = model.predict_proba(policy[columns])
    curve = cost_curve(predictions)
    # Resolve equal costs toward fewer contacts, then higher threshold.
    best = curve.sort_values(
        ["mean_cost", "contacts", "threshold"], ascending=[True, True, False]
    ).iloc[0]
    threshold = float(best.threshold)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "policy_predictions.parquet", index=False)
    curve.to_csv(output / "policy_cost_curve.csv", index=False)
    with (output / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    frozen = {
        "model": name,
        "feature_columns": columns,
        "params": run["best_params"],
        "costs": COSTS,
        "threshold": threshold if np.isfinite(threshold) else "never_contact",
        "selection": "maximum internal CV mean weekly ROC-AUC",
        "fit_last_cutoff": str(fit.cutoff.max()),
        "fit_rows": len(fit),
        "policy_first_cutoff": str(policy.cutoff.min()),
        "policy_last_cutoff": str(policy.cutoff.max()),
        "policy_rows": len(policy),
        "assumptions": "illustrative costs per customer-week; not incremental marketing ROI",
    }
    (output / "frozen_policy.json").write_text(json.dumps(frozen, indent=2))
    # Only after the model and threshold are frozen do we load final test.
    test = pd.read_parquet(data_dir / "test.parquet")
    if policy.cutoff.max() + pd.Timedelta(days=50) > test.cutoff.min():
        raise ValueError("Policy outcomes must mature before final test")
    test_predictions = test[["Customer ID", "cutoff", "RepurchaseWithin50Days"]].copy()
    probabilities = model.predict_proba(test[columns])
    test_predictions[["p_class_0", "p_class_1"]] = probabilities
    actions = (probabilities[:, 1] >= threshold).astype(int)
    test_predictions["contact"] = actions
    diagnostic = evaluate(test, probabilities)
    y = test.RepurchaseWithin50Days.to_numpy()
    loss = np.array(COSTS)[y, actions]
    report = {
        **frozen,
        "test_probability_and_05_metrics": diagnostic,
        "test_policy": {
            "mean_cost": float(loss.mean()),
            "total_cost": float(loss.sum()),
            "contacts": int(actions.sum()),
            "classification_report": classification_report(
                y, actions, labels=[0, 1], output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y, actions, labels=[0, 1]).tolist(),
            "accuracy": float((actions == y).mean()),
        },
        "test_contact_none_mean_cost": float(np.array(COSTS)[y, 0].mean()),
        "test_contact_all_mean_cost": float(np.array(COSTS)[y, 1].mean()),
    }
    test_predictions.to_parquet(output / "test_predictions.parquet", index=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, default=lambda value: value.item())
    )
    return report


def plot_policy_cost(output: Path):
    """Policy-holdout curve only; do not select thresholds from test outcomes."""
    curve = pd.read_csv(output / "policy_cost_curve.csv")
    report = json.loads((output / "report.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    curve.loc[np.isfinite(curve.threshold)].plot(
        x="threshold", y="mean_cost", ax=ax, label="Policy holdout"
    )
    ax.axhline(curve.iloc[0].mean_cost, label="Contact everyone", linestyle="--")
    ax.axhline(curve.iloc[-1].mean_cost, label="Contact nobody", linestyle=":")
    if isinstance(report["threshold"], (int, float)):
        ax.axvline(report["threshold"], label="Selected threshold", color="black")
    ax.set(xlabel="Purchase probability threshold", ylabel="Assumed cost (£ / customer-week)")
    ax.legend()
    return fig, ax
