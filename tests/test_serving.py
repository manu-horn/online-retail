import json
import pickle

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from online_retail.features.snapshots import customer_features
from online_retail.logging import configure_logging
from online_retail.serving.bundle import create_bundle, load_bundle
from online_retail.serving.inference import scoring_features
from online_retail.serving.workflow import run_scoring


@pytest.fixture
def scoring_setup(tmp_path):
    records = [
        ("customer-A", "2010-12-01"),
        ("customer-A", "2011-03-01"),
        ("customer-A", "2011-05-15"),
        ("customer-A", "2011-05-30"),
        ("customer-B", "2011-06-05"),
        ("future-C", "2011-06-06"),
    ]
    purchases = pd.DataFrame(
        [
            {
                "Customer ID": c,
                "Invoice": str(i),
                "InvoiceDate": pd.Timestamp(date),
                "StockCode": "A",
                "Quantity": 1,
                "Price": 10.0,
            }
            for i, (c, date) in enumerate(records)
        ]
    )
    features = customer_features(purchases, "2011-06-06")
    columns = [c for c in features if c not in ["Customer ID", "cutoff"]]
    x = pd.concat([features[columns]] * 25, ignore_index=True)
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(add_indicator=True, keep_empty_features=True)),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )
    pipeline.fit(x, [0] * 19 + [1] * 31)  # Exactly 0.62 purchase probability.
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.pkl").write_bytes(pickle.dumps(pipeline))
    (source / "frozen_policy.json").write_text(
        json.dumps(
            {
                "feature_columns": columns,
                "threshold": 0.62,
                "fit_last_cutoff": "2011-01-10",
                "policy_last_cutoff": "2011-04-11",
            }
        )
    )
    path = tmp_path / "purchases.parquet"
    purchases.to_parquet(path, index=False)
    bundle = tmp_path / "bundle"
    create_bundle(source, bundle)
    return purchases, path, source, bundle


def test_future_rows_do_not_change_features_or_eligibility(scoring_setup):
    purchases, _, _, _ = scoring_setup
    original = scoring_features(purchases, "2011-06-06", "2011-06-06")
    future = purchases.iloc[[-1]].copy()
    future["Quantity"] = 10000
    future["InvoiceDate"] = pd.Timestamp("2011-08-01")
    changed = scoring_features(pd.concat([purchases, future]), "2011-06-06", "2011-06-06")
    assert_frame_equal(original, changed)
    assert set(original["Customer ID"]) == {"customer-A", "customer-B"}
    assert pd.isna(original.set_index("Customer ID").loc["customer-B", "last_gap_days"])


def test_scoring_is_reproducible_versioned_and_boundary_inclusive(scoring_setup, tmp_path):
    _, path, source, bundle = scoring_setup
    before = (source / "model.pkl").read_bytes()
    logfile = tmp_path / "events.jsonl"
    logger = configure_logging(logfile)
    scores = run_scoring(path, bundle, "2011-06-06", "2011-06-06", tmp_path / "run1", logger)
    repeated = run_scoring(path, bundle, "2011-06-06", "2011-06-06", tmp_path / "run2", logger)
    assert np.allclose(scores.purchase_probability, 0.62) and scores.contact.all()
    assert_frame_equal(scores.drop(columns="run_id"), repeated.drop(columns="run_id"))
    assert set(["model_version", "policy_version", "feature_version", "data_version"]) <= set(
        scores
    )
    assert "RepurchaseWithin50Days" not in scores
    assert (source / "model.pkl").read_bytes() == before
    events = [json.loads(line) for line in logfile.read_text().splitlines()]
    assert sum(e["event"] == "scoring_succeeded" for e in events) == 2
    assert "customer-A" not in logfile.read_text()
    with pytest.raises(FileExistsError):
        run_scoring(path, bundle, "2011-06-06", "2011-06-06", tmp_path / "run1", logger)
    assert json.loads(logfile.read_text().splitlines()[-1])["event"] == "scoring_failed"


def test_bundle_integrity_checked_before_loading(scoring_setup):
    _, _, _, bundle = scoring_setup
    with (bundle / "model.pkl").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(bundle)


@pytest.mark.parametrize(
    "cutoff,coverage,message",
    [
        ("2011-06-07", "2011-06-08", "Monday"),
        ("2011-06-06 12:00", "2011-06-07", "Monday"),
        ("2011-06-13", "2011-06-06", "coverage"),
        ("2011-06-06T00:00:00Z", "2011-06-06", "timezone"),
    ],
)
def test_invalid_cutoffs_fail(scoring_setup, cutoff, coverage, message):
    purchases, _, _, _ = scoring_setup
    with pytest.raises(ValueError, match=message):
        scoring_features(purchases, cutoff, coverage)


def test_backdated_policy_and_bad_history_fail_without_outputs(scoring_setup, tmp_path):
    purchases, path, _, bundle = scoring_setup
    with pytest.raises(ValueError, match="label windows"):
        run_scoring(path, bundle, "2011-05-30", "2011-06-06", tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    purchases.loc[0, "Quantity"] = -1
    with pytest.raises(ValueError, match="positive"):
        scoring_features(purchases, "2011-06-06", "2011-06-06")


def test_policy_schema_must_match_pipeline(scoring_setup, tmp_path):
    _, _, source, _ = scoring_setup
    policy = json.loads((source / "frozen_policy.json").read_text())
    policy["feature_columns"] = policy["feature_columns"][::-1]
    (source / "frozen_policy.json").write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="schema"):
        create_bundle(source, tmp_path / "mismatch")
    assert not (tmp_path / "mismatch").exists()
