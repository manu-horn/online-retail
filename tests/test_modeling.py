import pandas as pd
import pytest

from online_retail.modeling.baseline import fit_baseline, logistic_baseline
from online_retail.modeling.metrics import evaluate
from online_retail.modeling.splits import temporal_folds


def test_temporal_folds_keep_dates_grouped_and_outcomes_mature():
    data = pd.DataFrame(
        [
            {"cutoff": date, "RepurchaseWithin50Days": label}
            for date in pd.date_range("2010-03-01", "2010-10-11", freq="W-MON")
            for label in (0, 1)
        ]
    )
    folds = list(temporal_folds(data))
    assert len(folds) == 3
    for fit_rows, valid_rows in folds:
        fit, valid = data.iloc[fit_rows], data.iloc[valid_rows]
        assert fit.cutoff.max() + pd.Timedelta(days=50) <= valid.cutoff.min()
        assert set(fit.cutoff).isdisjoint(valid.cutoff)
        assert fit.groupby("cutoff").size().eq(2).all()
        assert valid.groupby("cutoff").size().eq(2).all()


def test_preprocessing_is_training_only_and_baseline_is_unweighted():
    x = pd.DataFrame(
        {
            "recency_days": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "invoice_count": [1, 2, 3, 4, 5, 6],
            "last_gap_days": [None, 2.0, 3.0, None, 5.0, 6.0],
        }
    )
    model = fit_baseline(logistic_baseline(list(x)), x, [0, 1, 0, 1, 0, 1])
    imputer = model.named_steps["preprocess"].named_transformers_["linear"].named_steps["impute"]
    before = imputer.statistics_.copy()
    prediction = model.predict_proba(
        pd.DataFrame({"recency_days": [1000.0], "invoice_count": [500], "last_gap_days": [None]})
    )
    assert (imputer.statistics_ == before).all()
    assert prediction.shape == (1, 2)
    assert prediction.sum() == pytest.approx(1.0)
    assert model.named_steps["model"].class_weight is None
    assert model.named_steps["model"].C == 1.0


def test_probability_metrics_and_argmax_metrics_are_distinct():
    data = pd.DataFrame(
        {"cutoff": pd.to_datetime(["2011-01-03"] * 2), "RepurchaseWithin50Days": [0, 1]}
    )
    first = evaluate(data, [[0.8, 0.2], [0.2, 0.8]])
    second = evaluate(data, [[0.6, 0.4], [0.4, 0.6]])
    assert first["accuracy"] == second["accuracy"] == 1.0
    assert first["log_loss"] < second["log_loss"]
    assert first["brier_score"] == pytest.approx(0.04)
    assert first["brier_score"] < second["brier_score"]
