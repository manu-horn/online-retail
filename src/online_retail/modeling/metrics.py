"""Small probability-quality report for the initial baseline."""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score


def evaluate(data: pd.DataFrame, probabilities) -> dict:
    """Evaluate unchanged probabilities; labels use argmax only for diagnostics."""
    p = pd.DataFrame(probabilities, index=data.index, columns=[0, 1])
    truth, prediction = data.RepurchaseWithin50Days, p.idxmax(axis=1)
    weekly_losses = [
        log_loss(group.RepurchaseWithin50Days, p.loc[group.index], labels=[0, 1])
        for _, group in data.groupby("cutoff")
    ]
    return {
        "log_loss": log_loss(truth, p, labels=[0, 1]),
        "weekly_log_loss": float(pd.Series(weekly_losses).mean()),
        "brier_score": float(((p[1] - truth) ** 2).mean()),
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": recall_score(
            truth, prediction, labels=[0, 1], average="macro", zero_division=0
        ),
        "macro_f1": f1_score(truth, prediction, labels=[0, 1], average="macro", zero_division=0),
    }
