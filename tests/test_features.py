import pandas as pd
from pandas.testing import assert_frame_equal

from online_retail.features.rfm import RFMSegmenter
from online_retail.features.snapshots import (
    HORIZON,
    TARGET,
    build_snapshots,
    customer_features,
    partition_snapshots,
)


def purchases(records):
    return pd.DataFrame(
        [
            {
                "Customer ID": customer,
                "Invoice": str(i),
                "InvoiceDate": pd.Timestamp(date),
                "StockCode": "A",
                "Quantity": 2,
                "Price": 5.0,
            }
            for i, (customer, date) in enumerate(records)
        ]
    )


def test_features_are_past_only_sparse_and_count_invoices_not_lines():
    data = purchases(
        [
            ("A", "2010-01-01"),
            ("A", "2010-01-10"),
            ("A", "2010-02-01"),
            ("A", "2010-02-20"),
            ("B", "2010-02-28"),
            ("C", "2010-03-01"),
        ]
    )
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    result = customer_features(data, "2010-03-01").set_index("Customer ID")
    assert set(result.index) == {"A", "B"}
    assert result.loc["A", "invoice_count"] == 4
    assert result.loc["A", "monetary_total"] == 50
    assert result.loc["A", "average_invoice_value"] == 12.5
    assert result.loc["A", "invoice_count_30d"] == 2
    assert result.loc["A", "last_gap_days"] == 19
    assert result.loc["A", "gap_2_span_days"] == 41
    assert result.loc["A", "gap_3_span_days"] == 50
    assert pd.isna(result.loc["B", "last_gap_days"])
    assert not result.loc["B", "has_previous_purchase_day"]
    future = purchases([("A", "2010-04-01"), ("new", "2010-04-02")])
    assert_frame_equal(
        customer_features(data, "2010-03-01"),
        customer_features(pd.concat([data, future]), "2010-03-01"),
    )


def test_complete_label_boundaries_and_future_customer_eligibility():
    t = pd.Timestamp("2010-03-01")
    data = purchases(
        [
            ("early", "2010-02-01"),
            ("middle", "2010-02-01"),
            ("late", "2010-02-01"),
            ("never", "2010-02-01"),
            ("early", t + pd.Timedelta(days=20, hours=23)),
            ("middle", t + pd.Timedelta(days=21)),
            ("late", t + HORIZON),
            ("new", t),
            ("coverage", "2010-06-01"),
        ]
    )
    snapshots = build_snapshots(data)
    first = snapshots.loc[snapshots.cutoff.eq(t)].set_index("Customer ID")
    assert first[TARGET].to_dict() == {"early": 0, "late": 2, "middle": 1, "never": 2}
    assert (snapshots.cutoff + HORIZON <= pd.Timestamp("2010-06-01")).all()
    assert snapshots.cutoff.dt.dayofweek.eq(0).all()
    assert not snapshots.duplicated(["Customer ID", "cutoff"]).any()


def test_partition_maturity_and_disjoint_evaluation_dates():
    snapshots = pd.DataFrame({"cutoff": pd.date_range("2010-03-01", "2011-10-17", freq="W-MON")})
    parts = partition_snapshots(snapshots)
    assert (parts["train"].cutoff + HORIZON <= pd.Timestamp("2010-12-01")).all()
    assert parts["validation"].cutoff.min() == pd.Timestamp("2010-12-06")
    assert (parts["refit"].cutoff + HORIZON <= pd.Timestamp("2011-06-01")).all()
    assert parts["test"].cutoff.min() == pd.Timestamp("2011-06-06")
    for a, b in (("train", "validation"), ("validation", "test"), ("train", "test")):
        assert set(parts[a].cutoff).isdisjoint(parts[b].cutoff)


def test_rfm_order_and_validation_transform_does_not_refit():
    train = pd.DataFrame(
        {
            "recency_days": [1.0, 2.0, 40.0, 45.0, 200.0, 210.0],
            "invoice_count": [1, 1, 5, 6, 40, 50],
            "monetary_total": [10.0, 11.0, 100.0, 110.0, 1000.0, 1100.0],
        }
    )
    segmenter = RFMSegmenter({"r": 3, "f": 3, "m": 3}).fit(train)
    scored = segmenter.transform(train)
    assert scored.loc[0, "r_score"] == 100
    assert scored.loc[5, "r_score"] == 0
    assert scored.loc[0, "f_score"] == 0
    assert scored.loc[5, "m_score"] == 100
    centers = {k: v.cluster_centers_.copy() for k, v in segmenter.models_.items()}
    segmenter.transform(train * 100)
    for key, model in segmenter.models_.items():
        assert (centers[key] == model.cluster_centers_).all()


def test_component_normalization_with_different_cluster_counts():
    from itertools import product

    import numpy as np

    values = pd.DataFrame(
        list(product([1.0, 100.0], [1, 10, 100], [1.0, 10.0, 100.0, 1000.0])),
        columns=["recency_days", "invoice_count", "monetary_total"],
    )
    model = RFMSegmenter({"r": 2, "f": 3, "m": 4}).fit(values)
    scored = model.transform(values)
    for name in ("r", "f", "m"):
        assert scored[f"{name}_score"].min() == 0
        assert scored[f"{name}_score"].max() == 100
    assert np.allclose(scored.rfm_score, scored[["r_score", "f_score", "m_score"]].mean(axis=1))
    edges = list(model.segment_edges_)
    model.transform(values * 2)
    assert model.segment_edges_ == edges


def test_rfm_diagnostics_are_training_only_and_profiles_use_one_cutoff():
    import pytest

    from online_retail.analysis.rfm import cluster_diagnostics, recommend_clusters, segment_profiles

    values = pd.DataFrame(
        {
            "recency_days": [1.0, 2.0, 3.0, 40.0, 45.0, 50.0, 200.0, 210.0, 220.0],
            "invoice_count": [1, 2, 3, 5, 6, 7, 40, 50, 60],
            "monetary_total": [10.0, 11.0, 12.0, 100.0, 110.0, 120.0, 1000.0, 1100.0, 1200.0],
        }
    )
    diagnostics = cluster_diagnostics(values, max_k=4)
    selected = recommend_clusters(diagnostics)
    assert set(selected) == {"r", "f", "m"}
    assert all(2 <= k <= 3 for k in selected.values())
    for _, group in diagnostics.groupby("component"):
        assert group.inertia.is_monotonic_decreasing
    with pytest.raises(ValueError, match="one cutoff"):
        segment_profiles(pd.DataFrame({"cutoff": [1, 2], "Customer ID": ["A", "A"]}))
