"""Read-only evaluation of frozen probabilities and contact decisions."""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

KEYS = ["Customer ID", "cutoff"]
TARGET = "RepurchaseWithin50Days"


def load_evaluation(data_dir: Path, artifact_dir: Path):
    """Verify saved predictions against exact frozen pipeline; never fit anything."""
    frozen = json.loads((artifact_dir / "frozen_policy.json").read_text())
    saved = pd.read_parquet(artifact_dir / "test_predictions.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    if saved.duplicated(KEYS).any() or test.duplicated(KEYS).any():
        raise ValueError("Duplicate customer-cutoff keys")
    merged = test.merge(
        saved[KEYS + [TARGET, "p_class_0", "p_class_1", "contact"]],
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_saved"),
    )
    if not merged._merge.eq("both").all() or not merged[TARGET].eq(merged[TARGET + "_saved"]).all():
        raise ValueError("Saved predictions and test population/outcomes disagree")
    # Trusted local artifact only. Imputation is included in the fitted pipeline.
    with (artifact_dir / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    probability = model.predict_proba(merged[frozen["feature_columns"]])
    if not np.allclose(probability, merged[["p_class_0", "p_class_1"]], atol=1e-7, rtol=1e-7):
        raise ValueError("Saved probabilities disagree with frozen pipeline")
    if not merged.contact.eq(merged.p_class_1.ge(frozen["threshold"])).all():
        raise ValueError("Saved actions disagree with frozen threshold")
    refit = pd.read_parquet(data_dir / "refit.parquet")
    fitting = refit.loc[refit.cutoff.le(pd.Timestamp(frozen["fit_last_cutoff"]))]
    if len(fitting) != frozen["fit_rows"]:
        raise ValueError("Cannot reconstruct final fitting population")
    merged["history_group"] = pd.cut(
        merged.invoice_count, [0, 1, 3, np.inf], labels=["1 invoice", "2–3 invoices", "4+ invoices"]
    )
    merged["identity_group"] = np.where(
        merged["Customer ID"].isin(fitting["Customer ID"]),
        "Seen in final fit",
        "Unseen in final fit",
    )
    return merged.drop(columns=["_merge", TARGET + "_saved"]), frozen


def summarize(data, threshold, costs):
    """Metrics for one population; absent-class ROC/AP/recall are undefined."""
    y, p = data[TARGET], data.p_class_1
    action = p.ge(threshold)
    tn, fp, fn, tp = confusion_matrix(y, action, labels=[0, 1]).ravel()
    loss = np.asarray(costs)[y.to_numpy(dtype=int), action.to_numpy(dtype=int)]
    return {
        "snapshots": len(data),
        "customers": data["Customer ID"].nunique(),
        "buyers": int(y.sum()),
        "nonbuyers": int(y.eq(0).sum()),
        "purchase_rate": y.mean(),
        "mean_probability": p.mean(),
        "roc_auc": roc_auc_score(y, p) if y.nunique() == 2 else np.nan,
        "average_precision": average_precision_score(y, p) if y.eq(1).any() else np.nan,
        "log_loss": log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1]),
        "brier_score": ((p - y) ** 2).mean(),
        "accuracy": accuracy_score(y, action),
        "precision": precision_score(y, action, zero_division=0) if action.any() else np.nan,
        "recall": recall_score(y, action, zero_division=0) if y.eq(1).any() else np.nan,
        "f1": f1_score(y, action, zero_division=0),
        "contact_rate": action.mean(),
        "contacts": int(action.sum()),
        "mean_cost": loss.mean(),
        "total_cost": loss.sum(),
        "contact_none_mean_cost": np.asarray(costs)[y.to_numpy(dtype=int), 0].mean(),
        "contact_all_mean_cost": np.asarray(costs)[y.to_numpy(dtype=int), 1].mean(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def grouped_metrics(data, group, threshold, costs):
    rows = [
        {group: name, **summarize(part, threshold, costs)}
        for name, part in data.groupby(group, observed=True, sort=True)
    ]
    return pd.DataFrame(rows).set_index(group)


def plot_overall(data, threshold):
    y, p = data[TARGET], data.p_class_1
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    RocCurveDisplay.from_predictions(y, p, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    PrecisionRecallDisplay.from_predictions(y, p, ax=axes[1])
    axes[1].axhline(y.mean(), color="black", linestyle="--", label="Purchase prevalence")
    axes[1].legend()
    CalibrationDisplay.from_predictions(y, p, n_bins=10, strategy="quantile", ax=axes[2])
    axes[2].set_title("Test reliability (diagnostic only)")
    fig2, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        y, p.ge(threshold), labels=[0, 1], ax=ax, colorbar=False, values_format="d"
    )
    ax.set_title(f"Frozen threshold {threshold:.2f}")
    return fig, fig2


def plot_weekly(weekly):
    panels = [
        (["purchase_rate", "mean_probability"], "Purchase rate and probability"),
        (["roc_auc", "average_precision"], "Ranking quality"),
        (["precision", "recall"], "Policy precision and recall"),
        (["contact_rate"], "Contact rate"),
        (
            ["mean_cost", "contact_none_mean_cost", "contact_all_mean_cost"],
            "Assumed cost (£ / customer-week)",
        ),
        (["snapshots", "buyers"], "Weekly denominators"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, layout="constrained")
    for ax, (columns, title) in zip(axes.flat, panels, strict=True):
        weekly[columns].plot(ax=ax)
        ax.set(title=title, xlabel="Scoring cutoff")
        ax.grid(alpha=0.2)
    return fig, axes


def plot_groups(table, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for ax, columns in zip(
        axes,
        [
            ["roc_auc", "average_precision"],
            ["precision", "recall"],
            ["purchase_rate", "contact_rate"],
        ],
        strict=True,
    ):
        table[columns].plot.bar(ax=ax, rot=15)
        ax.set(ylim=(0, 1), xlabel="", ylabel="Score / fraction")
    fig.suptitle(title)
    return fig, axes
