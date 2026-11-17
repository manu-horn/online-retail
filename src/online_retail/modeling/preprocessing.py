"""Fold-fitted transformations for logistic models; source datasets stay unchanged."""

import math

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def log1p_values(values):
    """Top-level function keeps saved pipelines serializable."""
    return pd.DataFrame(values).map(math.log1p).to_numpy()


def logistic_preprocessor(columns: list[str]) -> ColumnTransformer:
    logarithmic = [
        c
        for c in columns
        if any(term in c for term in ("count", "monetary", "average_", "distinct_product"))
    ]
    linear = [c for c in columns if c not in logarithmic]
    return ColumnTransformer(
        [
            (
                "log",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="median", add_indicator=True, keep_empty_features=True
                            ),
                        ),
                        (
                            "log1p",
                            FunctionTransformer(log1p_values, feature_names_out="one-to-one"),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                logarithmic,
            ),
            (
                "linear",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="median", add_indicator=True, keep_empty_features=True
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                linear,
            ),
        ]
    )
