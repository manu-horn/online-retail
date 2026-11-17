"""Evaluate a specified binary policy; no business costs are assumed."""

import numpy as np


def campaign_cost(y_true, purchase_probability, threshold: float, costs) -> dict:
    """Costs[actual outcome, contact action]; contact when probability >= threshold.

    Row 0 means no purchase, row 1 purchase. Column 0 means no contact,
    column 1 contact. This is a descriptive decision loss, not causal campaign ROI.
    """
    y = np.asarray(y_true)
    p = np.asarray(purchase_probability, dtype=float)
    matrix = np.asarray(costs, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError(
            "Provide equally sized nonempty one-dimensional outcomes and probabilities"
        )
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary outcomes and finite probabilities in [0, 1]")
    if not 0 <= threshold <= 1 or matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError("Expected threshold in [0, 1] and finite 2x2 cost matrix")
    action = (p >= threshold).astype(int)
    loss = matrix[y.astype(int), action]
    return {
        "total_cost": float(loss.sum()),
        "mean_cost": float(loss.mean()),
        "contacts": int(action.sum()),
    }
