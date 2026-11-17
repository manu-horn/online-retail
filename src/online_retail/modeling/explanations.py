"""Standard SHAP TreeExplainer for the frozen model, in log-odds units."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap


def explain_test(data, columns, model_path: Path, max_rows=5000, seed=42):
    """Uniform snapshot sample, not outcome-balanced; keep imputation indicators."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    sample = data.sample(n=min(max_rows, len(data)), random_state=seed).copy()
    # Load only a trusted local fitted pipeline; never fit the imputer or booster.
    with model_path.open("rb") as handle:
        pipeline = pickle.load(handle)
    transformed = pipeline.named_steps["impute"].transform(sample[columns])
    names = pipeline.named_steps["impute"].get_feature_names_out(columns)
    explainer = shap.TreeExplainer(
        pipeline.named_steps["model"],
        model_output="raw",
        feature_perturbation="tree_path_dependent",
    )
    explanation = explainer(transformed, check_additivity=True)
    contribution = explanation.values
    base_values = np.broadcast_to(explanation.base_values, (len(sample),))
    margin = pipeline.named_steps["model"].predict(transformed, output_margin=True)
    if not np.allclose(contribution.sum(axis=1) + base_values, margin, atol=1e-4, rtol=1e-5):
        raise ValueError("SHAP additivity does not reproduce prediction margins")
    probability = 1 / (1 + np.exp(-(contribution.sum(axis=1) + base_values)))
    if not np.allclose(probability, pipeline.predict_proba(sample[columns])[:, 1], atol=1e-5):
        raise ValueError("SHAP margins do not reproduce frozen probabilities")
    values = pd.DataFrame(contribution, columns=names, index=sample.index)
    features = pd.DataFrame(transformed, columns=names, index=sample.index)
    base = pd.Series(base_values, index=sample.index, name="base_log_odds")
    return sample, values, features, base


def importance_summary(values, sample):
    """Mean absolute magnitude and signed direction; all rows count equally."""
    overall = pd.DataFrame(
        {"mean_abs_shap": values.abs().mean(), "mean_shap": values.mean()}
    ).sort_values("mean_abs_shap", ascending=False)
    groups = pd.DataFrame(
        {
            str(name): values.loc[group.index].abs().mean()
            for name, group in sample.groupby("history_group", observed=True)
        }
    )
    return overall.rename_axis("feature"), groups.reindex(overall.index).rename_axis("feature")
