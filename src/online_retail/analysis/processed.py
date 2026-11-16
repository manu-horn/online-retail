"""Training-set feature diagnostics; never fit models or select features automatically."""

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogLocator, StrMethodFormatter

CLASS_NAMES = {0: "0: days 0–20", 1: "1: days 21–49", 2: "2: no purchase within 50 days"}


def feature_summary(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Include missingness, cardinality and full numeric ranges without dropping tails."""
    features = data[columns]
    result = pd.DataFrame(
        {
            "dtype": features.dtypes.astype(str),
            "missing_pct": features.isna().mean() * 100,
            "unique_values": features.nunique(),
        }
    )
    numeric = features.select_dtypes("number")
    stats = numeric.describe(percentiles=[0.5, 0.95, 0.99]).T
    return result.join(stats[["min", "50%", "95%", "99%", "max"]])


def class_shares(data: pd.DataFrame, target: str, group: str) -> pd.DataFrame:
    """Conditional class percentages within each group, preserving absent classes."""
    counts = pd.crosstab(data[group], data[target]).reindex(columns=[0, 1, 2], fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0) * 100


def plot_target_over_time(data: pd.DataFrame, target: str):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    shares = class_shares(data, target, "cutoff")
    shares.rename(columns=CLASS_NAMES).plot(ax=axes[0])
    axes[0].set(
        ylabel="Within-week class share (%)", ylim=(0, 100), title="Training outcome balance"
    )
    data.groupby("cutoff").size().plot(ax=axes[1])
    axes[1].set(ylabel="Eligible customers", xlabel="Scoring cutoff")
    return fig, axes


def plot_feature_distributions(data: pd.DataFrame, columns: list[str]):
    """Original-unit histograms: base-10 spend/frequency axes, linear days/zero counts."""
    numeric = data[columns].select_dtypes("number")
    fig, axes = plt.subplots(
        math.ceil(len(numeric.columns) / 3),
        3,
        figsize=(15, math.ceil(len(numeric.columns) / 3) * 2.7),
        layout="constrained",
    )
    for ax, column in zip(axes.flat, numeric.columns, strict=False):
        values = numeric[column].dropna()
        xlabel = "Value"
        bins = 30
        if column in {"invoice_count", "monetary_total"} and not values.empty:
            positive = values.loc[values.gt(0)]
            if not positive.empty:
                low, high = math.log10(positive.min()), math.log10(positive.max())
                if low == high:
                    low, high = low - 0.1, high + 0.1
                bins = [10 ** (low + (high - low) * i / 30) for i in range(31)]
                # Include the exact extrema despite floating-point exponent rounding.
                bins[0] = min(bins[0], positive.min())
                bins[-1] = max(bins[-1], positive.max())
                ax.hist(positive, bins=bins, edgecolor="white")
                ax.set_xscale("log", base=10)
                ax.xaxis.set_major_locator(LogLocator(base=10))
                ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
                xlabel = "Value (base-10 log axis, original units)"
                excluded = values.le(0).sum()
                if excluded:
                    ax.text(
                        0.98,
                        0.95,
                        f"Nonpositive: {excluded:,} (not on log axis)",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8,
                    )
            else:
                ax.hist(values, bins=bins, edgecolor="white")
        else:
            ax.hist(values, bins=bins, edgecolor="white")
        ax.set(title=column, ylabel="Customers", xlabel=xlabel)
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(numeric.columns) :]:
        ax.set_visible(False)
    return fig, axes


def plot_correlations(data: pd.DataFrame, columns: list[str]):
    """Predictor-only Spearman correlation; class IDs are not continuous outcomes."""
    numeric = data[columns].select_dtypes("number")
    # Constant calendar features at a single cutoff have undefined correlation.
    numeric = numeric.loc[:, numeric.nunique() > 1]
    correlation = numeric.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    sns.heatmap(
        correlation,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
    )
    ax.set_title("Training predictor Spearman correlations (pairwise complete observations)")
    return fig, ax


def plot_group_outcomes(data: pd.DataFrame, target: str):
    """Show historical associations, not validated predictions or causal effects."""
    groups = data.copy()
    groups["history_group"] = pd.cut(
        groups.invoice_count,
        bins=[0, 1, 3, float("inf")],
        labels=["1 invoice", "2–3 invoices", "4+ invoices"],
    )
    groups["recency_group"] = pd.cut(
        groups.recency_days,
        bins=[0, 21, 50, 90, float("inf")],
        right=False,
        labels=["<21 days", "21–49 days", "50–89 days", "90+ days"],
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, group in zip(axes, ["history_group", "recency_group", "rfm_segment"], strict=True):
        shares = class_shares(groups, target, group)
        if group == "rfm_segment":
            shares = shares.reindex(["low", "medium", "high"])
        shares.rename(columns=CLASS_NAMES).plot.bar(stacked=True, ax=ax, legend=False)
        ax.set(title=group, ylabel="Customer-snapshot share (%)", ylim=(0, 100))
        ax.tick_params(axis="x", rotation=30)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3)
    return fig, axes


def plot_weekly_feature_medians(data: pd.DataFrame):
    columns = ["recency_days", "invoice_count", "monetary_total", "observed_history_days"]
    medians = data.groupby("cutoff")[columns].median()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, column in zip(axes.flat, columns, strict=True):
        medians[column].plot(ax=ax)
        ax.set(title=f"Weekly median: {column}", xlabel="Cutoff")
        ax.grid(alpha=0.2)
    return fig, axes


def target_correlations(data: pd.DataFrame, columns: list[str], target: str) -> pd.DataFrame:
    """Spearman association with one binary indicator per outcome, pairwise complete.

    No ordinal interpretation of class IDs and no independence-based p-values.
    """
    numeric = data[columns].select_dtypes("number")
    indicators = pd.DataFrame(
        {label: data[target].eq(value).astype(float) for value, label in CLASS_NAMES.items()},
        index=data.index,
    )
    return pd.DataFrame(
        {
            label: numeric.apply(
                lambda values, indicator=indicators[label]: values.corr(
                    indicator, method="spearman"
                )
            )
            for label in indicators
        }
    )


def plot_target_correlations(correlation: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, center=0, ax=ax)
    ax.set(title="Training features vs each outcome (Spearman)", xlabel="Outcome indicator")
    ax.tick_params(axis="x", rotation=15)
    return fig, ax
