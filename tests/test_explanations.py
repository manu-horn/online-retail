import pickle

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from online_retail.modeling.explanations import explain_test, importance_summary


def test_shap_library_preserves_sample_alignment_and_missing_indicators(tmp_path):
    data = pd.DataFrame(
        {
            "x": np.arange(100, dtype=float),
            "gap": [np.nan, 2.0] * 50,
            "history_group": ["one", "many"] * 50,
        }
    )
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(add_indicator=True)),
            ("model", XGBClassifier(n_estimators=5, max_depth=2, n_jobs=1)),
        ]
    )
    columns = ["x", "gap"]
    pipeline.fit(data[columns], data.x.gt(50).astype(int))
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(pipeline))
    before = path.read_bytes()
    sample, values, features, base = explain_test(data, columns, path, max_rows=30)
    assert len(sample) == 30
    assert values.index.equals(sample.index) and features.index.equals(sample.index)
    assert "missingindicator_gap" in values
    reconstructed = 1 / (1 + np.exp(-(values.sum(axis=1) + base)))
    assert np.allclose(reconstructed, pipeline.predict_proba(sample[columns])[:, 1], atol=1e-5)
    overall, groups = importance_summary(values, sample)
    assert set(groups.columns) == {"one", "many"}
    assert overall.mean_abs_shap.is_monotonic_decreasing
    assert path.read_bytes() == before
