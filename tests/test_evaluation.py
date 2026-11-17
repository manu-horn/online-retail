import numpy as np
import pandas as pd
import pytest

from online_retail.modeling.evaluation import grouped_metrics, summarize


def test_frozen_threshold_counts_costs_and_group_denominators():
    data = pd.DataFrame(
        {
            "Customer ID": ["a", "b", "a", "c"],
            "RepurchaseWithin50Days": [0, 0, 1, 1],
            "p_class_1": [0.1, 0.62, 0.3, 0.9],
            "group": ["x", "x", "y", "y"],
        }
    )
    result = summarize(data, 0.62, [[0, 15], [25, 10]])
    assert [result[k] for k in ("tn", "fp", "fn", "tp")] == [1, 1, 1, 1]
    assert result["mean_cost"] == 12.5
    assert result["customers"] == 3
    grouped = grouped_metrics(data, "group", 0.62, [[0, 15], [25, 10]])
    assert grouped.snapshots.sum() == 4
    assert np.isnan(grouped.loc["x", "roc_auc"])
    assert grouped.loc["y", "recall"] == pytest.approx(0.5)


def test_undefined_slice_metrics_are_not_misleading_zeroes():
    data = pd.DataFrame({"Customer ID": ["a"], "RepurchaseWithin50Days": [0], "p_class_1": [0.1]})
    result = summarize(data, 0.62, [[0, 15], [25, 10]])
    for key in ("roc_auc", "average_precision", "precision", "recall"):
        assert np.isnan(result[key])
    assert result["mean_cost"] == 0
