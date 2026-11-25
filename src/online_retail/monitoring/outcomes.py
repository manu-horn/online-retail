"""Evaluate a saved scored population only after complete 50-day follow-up."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from online_retail.monitoring.core import HISTORY_GROUPS, IDENTITIES, history_fingerprint
from online_retail.serving.bundle import checksum


def _metrics(data):
    y, p, action = data.target, data.purchase_probability, data.contact.astype(int)
    return {
        "snapshots": len(data),
        "buyers": int(y.sum()),
        "buyer_share": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if y.any() else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(np.mean((p - y) ** 2)),
        "accuracy": float(accuracy_score(y, action)),
        "buyer_precision": float(precision_score(y, action, zero_division=0)),
        "buyer_recall": float(recall_score(y, action, zero_division=0)),
        "buyer_f1": float(f1_score(y, action, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, action, labels=[0, 1]).tolist(),
    }


def evaluate_outcomes(run_dir, purchases_path, coverage_end, output):
    run_dir, purchases_path, output = map(Path, (run_dir, purchases_path, output))
    if output.exists():
        raise FileExistsError("Outcome output already exists")
    metadata = json.loads((run_dir / "run.json").read_text())
    cutoff, coverage_end = pd.Timestamp(metadata["cutoff"]), pd.Timestamp(coverage_end)
    window_end = cutoff + pd.Timedelta(days=50)
    if pd.isna(coverage_end) or coverage_end.tzinfo is not None or coverage_end < window_end:
        raise ValueError("Complete 50-day outcome coverage is required before evaluation")
    scores = pd.read_parquet(run_dir / "scores.parquet")
    if scores.empty or scores.duplicated(["Customer ID", "cutoff"]).any():
        raise ValueError("Saved scored population must have unique nonempty keys")
    if scores[["Customer ID", "cutoff"]].isna().any().any() or not scores.cutoff.eq(cutoff).all():
        raise ValueError("Saved scoring dates do not match run metadata")
    if len(scores) != metadata["scored_customers"]:
        raise ValueError("Saved population count does not match run metadata")
    for key in (*IDENTITIES, "run_id", "data_version"):
        if not scores[key].eq(metadata[key]).all():
            raise ValueError("Saved score identities do not match run metadata")
    if "history_group" not in scores or not scores.history_group.isin(HISTORY_GROUPS).all():
        raise ValueError("Use a monitored scoring run with saved history groups")
    p = scores.purchase_probability
    if not np.isfinite(p).all() or not p.between(0, 1).all():
        raise ValueError("Saved probabilities are invalid")
    if (
        not scores.contact.isin([True, False]).all()
        or not scores.contact.eq(p.ge(metadata["threshold"])).all()
    ):
        raise ValueError("Saved decisions do not match the frozen threshold")
    purchases = pd.read_parquet(purchases_path)
    if (
        not pd.api.types.is_datetime64_any_dtype(purchases.InvoiceDate)
        or purchases.InvoiceDate.dt.tz is not None
        or purchases.InvoiceDate.isna().any()
    ):
        raise ValueError("Follow-up dates must be present and timezone-naive")
    if history_fingerprint(purchases, cutoff) != metadata["history_sha256"]:
        raise ValueError(
            "Pre-cutoff history changed; reconcile backfills before evaluating this run"
        )
    future = purchases.loc[purchases.InvoiceDate.ge(cutoff) & purchases.InvoiceDate.lt(window_end)]
    required = ["Customer ID", "Invoice", "StockCode", "Quantity", "Price"]
    if future[required].isna().any().any():
        raise ValueError("Follow-up must contain cleaned identified purchases")
    for name in ("Customer ID", "Invoice", "StockCode"):
        if not pd.api.types.is_string_dtype(future[name]) or future[name].str.strip().eq("").any():
            raise ValueError("Follow-up identifiers must be nonempty strings")
    from online_retail.data.preprocess import EXCLUDED_STOCK_CODES

    if (
        future.Invoice.str.upper().str.startswith("C").any()
        or future.StockCode.str.upper().isin(EXCLUDED_STOCK_CODES).any()
    ):
        raise ValueError("Clean cancellations/adjustments before outcome evaluation")
    for name in ("Quantity", "Price"):
        if (
            not pd.api.types.is_numeric_dtype(future[name])
            or not np.isfinite(future[name]).all()
            or future[name].le(0).any()
        ):
            raise ValueError("Follow-up must contain positive finite purchases")
    if future.Quantity.mod(1).ne(0).any():
        raise ValueError("Follow-up quantities must be whole counts")
    scores["target"] = scores["Customer ID"].isin(future["Customer ID"]).astype(int)
    report = {
        "schema_version": 1,
        "purpose": "delayed descriptive evaluation; no model or policy selection",
        "run_id": metadata["run_id"],
        "cutoff": str(cutoff),
        "outcome_end_exclusive": str(window_end),
        "coverage_end_exclusive": str(coverage_end),
        "threshold": metadata["threshold"],
        **{key: metadata[key] for key in IDENTITIES},
        "scores_sha256": checksum(run_dir / "scores.parquet"),
        "outcome_data_sha256": checksum(purchases_path),
        "history_unchanged": True,
        "overall": _metrics(scores),
        "history_groups": {
            str(name): _metrics(group) for name, group in scores.groupby("history_group")
        },
    }
    lines = [
        "# Mature-outcome monitoring",
        "",
        f"Scoring cutoff: {cutoff}; outcomes: [{cutoff}, {window_end}).",
        "",
        "These are observed purchase outcomes, not incremental campaign effects.",
        "",
        "| Population | Snapshots | ROC-AUC | Buyer precision | Buyer recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in {"overall": report["overall"], **report["history_groups"]}.items():
        auc = "undefined" if values["roc_auc"] is None else f"{values['roc_auc']:.3f}"
        lines.append(
            f"| {name} | {values['snapshots']} | {auc} | {values['buyer_precision']:.3f} "
            f"| {values['buyer_recall']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Confusion matrix order: [[TN, FP], [FN, TP]].",
            "",
            json.dumps(report["overall"]["confusion_matrix"]),
            "",
            "Only scored customers are evaluated. Each row is one customer at this cutoff.",
            "Weekly populations overlap; repeated weeks are not independent samples.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "outcomes"
        stage.mkdir()
        (stage / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (stage / "report.md").write_text("\n".join(lines) + "\n")
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return report
