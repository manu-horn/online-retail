"""Small probability-quality report for the initial baseline."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(data: pd.DataFrame, probabilities) -> dict:
    """Evaluate unchanged probabilities; labels use argmax only for diagnostics."""
    p = pd.DataFrame(probabilities, index=data.index, columns=[0, 1])
    truth, prediction = data.RepurchaseWithin50Days, p.idxmax(axis=1)
    weekly_losses = [
        log_loss(group.RepurchaseWithin50Days, p.loc[group.index], labels=[0, 1])
        for _, group in data.groupby("cutoff")
    ]
    result = {
        "log_loss": log_loss(truth, p, labels=[0, 1]),
        "weekly_log_loss": float(pd.Series(weekly_losses).mean()),
        "brier_score": float(((p[1] - truth) ** 2).mean()),
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": recall_score(
            truth, prediction, labels=[0, 1], average="macro", zero_division=0
        ),
        "macro_f1": f1_score(truth, prediction, labels=[0, 1], average="macro", zero_division=0),
    }

    result["roc_auc"] = roc_auc_score(truth, p[1]) if truth.nunique() == 2 else np.nan
    result["average_precision"] = (
        average_precision_score(truth, p[1]) if truth.eq(1).any() else np.nan
    )
    for average in ("macro", "micro", "weighted"):
        for name, metric in (
            ("precision", precision_score),
            ("recall", recall_score),
            ("f1", f1_score),
        ):
            result[f"{average}_{name}"] = metric(
                truth, prediction, average=average, zero_division=0
            )
        result[f"{average}_f2"] = fbeta_score(
            truth, prediction, beta=2, average=average, zero_division=0
        )
    for k in (0, 1):
        for name, metric in (
            ("precision", precision_score),
            ("recall", recall_score),
            ("f1", f1_score),
        ):
            result[f"class_{k}_{name}"] = metric(
                truth, prediction, labels=[k], average=None, zero_division=0
            )[0]
    result.update(
        zip(
            ("tn", "fp", "fn", "tp"),
            confusion_matrix(truth, prediction, labels=[0, 1]).ravel(),
            strict=True,
        )
    )
    return result
