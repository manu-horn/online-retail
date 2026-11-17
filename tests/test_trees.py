import numpy as np
import optuna
import pandas as pd
import pytest

from online_retail.modeling.baseline import fit_baseline
from online_retail.modeling.trees import tree_pipeline


@pytest.mark.parametrize("name", ["lightgbm", "xgboost", "random_forest"])
def test_tree_pipeline_preserves_fit_imputation_and_binary_probabilities(name):
    x = pd.DataFrame(
        {"recency_days": np.arange(200, dtype=float), "last_gap_days": [np.nan, 5.0] * 100}
    )
    trial = optuna.trial.FixedTrial(
        {
            "n_estimators": 100,
            "max_depth": 3,
            "min_samples_leaf": 20,
            "max_features": 0.8,
            "max_samples": 0.8,
            "learning_rate": 0.05,
            "reg_lambda": 1.0,
            "reg_alpha": 0.1,
            "colsample_bytree": 0.8,
            "num_leaves": 7,
            "min_child_samples": 30,
            "min_child_weight": 5.0,
            "subsample": 0.8,
        }
    )
    model = fit_baseline(tree_pipeline(name, trial), x, [0] * 100 + [1] * 100)
    statistics = model.named_steps["impute"].statistics_.copy()
    probabilities = model.predict_proba(x.iloc[[0, 199]])
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert np.array_equal(model.named_steps["impute"].statistics_, statistics)
    assert probabilities[0, 1] < probabilities[1, 1]
