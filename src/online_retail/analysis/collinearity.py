"""Descriptive training-only redundancy checks, not automatic feature selection."""

import math
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def correlated_pairs(data: pd.DataFrame, columns: list[str], threshold=0.8) -> pd.DataFrame:
    numeric = data[columns].select_dtypes("number")
    pearson = numeric.corr()
    spearman = numeric.corr(method="spearman")
    records = []
    for a, b in combinations(numeric.columns, 2):
        if max(abs(pearson.loc[a, b]), abs(spearman.loc[a, b])) >= threshold:
            records.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "pearson": pearson.loc[a, b],
                    "spearman": spearman.loc[a, b],
                }
            )
    return pd.DataFrame(records, columns=["feature_a", "feature_b", "pearson", "spearman"])


def encoded_vif(data: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """VIF on an intercept-adjusted, standardized diagnostic design matrix.

    Median-fill numeric columns and reference-code string/category/bool columns.
    This is EDA only; model preprocessing must be fitted again within each fold.
    """
    frame = data[columns].copy()
    categorical = [
        c
        for c in frame
        if not pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c])
    ]
    for column in categorical:
        frame[column] = frame[column].astype("string").fillna("<missing>")
    numeric = [c for c in frame if c not in categorical]
    frame[numeric] = frame[numeric].fillna(frame[numeric].median())
    design = pd.get_dummies(frame, columns=categorical, drop_first=True, dtype=float)
    excluded = design.columns[design.nunique(dropna=False).le(1) | design.isna().any()].tolist()
    design = design.drop(columns=excluded)
    scaled = pd.DataFrame(StandardScaler().fit_transform(design), columns=design.columns)
    records = []
    for column in scaled:
        others = scaled.drop(columns=column)
        r2 = (
            LinearRegression().fit(others, scaled[column]).score(others, scaled[column])
            if len(others.columns)
            else 0.0
        )
        vif = float("inf") if 1 - r2 <= 1e-10 else 1 / (1 - r2)
        records.append({"feature": column, "vif": vif, "r_squared_other_features": r2})
    return pd.DataFrame(records).sort_values("vif", ascending=False), excluded


def categorical_associations(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Bias-corrected Cramer's V (0–1); no independent-sample p-values."""
    result = pd.DataFrame(float("nan"), index=columns, columns=columns)
    for a in columns:
        result.loc[a, a] = 1.0
    for a, b in combinations(columns, 2):
        table = pd.crosstab(
            data[a].astype("string").fillna("<missing>"),
            data[b].astype("string").fillna("<missing>"),
        )
        n = table.to_numpy().sum()
        r, k = table.shape
        if n <= 1 or min(r, k) <= 1:
            continue
        expected = pd.DataFrame(
            [[row * col / n for col in table.sum(axis=0)] for row in table.sum(axis=1)],
            index=table.index,
            columns=table.columns,
        )
        phi2 = ((table - expected) ** 2 / expected).to_numpy().sum() / n
        corrected = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
        denominator = min(k - (k - 1) ** 2 / (n - 1) - 1, r - (r - 1) ** 2 / (n - 1) - 1)
        value = math.sqrt(corrected / denominator) if denominator > 0 else float("nan")
        result.loc[a, b] = result.loc[b, a] = value
    return result


def mixed_associations(
    data: pd.DataFrame, categorical: list[str], numeric: list[str]
) -> pd.DataFrame:
    """Correlation ratio eta: categorical grouping explains numeric variation.

    Univariate and directional; high eta does not prove a linear dependency.
    """
    result = pd.DataFrame(index=categorical, columns=numeric, dtype=float)
    for cat in categorical:
        for number in numeric:
            pairs = data[[cat, number]].dropna()
            values = pairs[number].astype(float)
            total = ((values - values.mean()) ** 2).sum()
            groups = values.groupby(pairs[cat], observed=True).agg(["size", "mean"])
            between = (groups["size"] * (groups["mean"] - values.mean()) ** 2).sum()
            result.loc[cat, number] = (
                math.sqrt(min(1.0, max(0.0, between / total))) if total else float("nan")
            )
    return result


def plot_associations(matrix: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(max(7, len(matrix.columns) * 1.1), 5), layout="constrained")
    sns.heatmap(matrix, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="Blues", ax=ax)
    ax.set_title(title)
    return fig, ax
