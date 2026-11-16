"""Training-only diagnostics for separate R, F, and M cluster counts."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from online_retail.features.rfm import RFM, component_values


def cluster_diagnostics(training: pd.DataFrame, max_k: int = 6) -> pd.DataFrame:
    """Inertia on all train rows; silhouette on a fixed sample of at most 1500."""
    records = []
    for component, column in RFM.items():
        values = component_values(training, component)
        sample = values.sample(min(1500, len(values)), random_state=42)
        for k in range(1, min(max_k, values[column].nunique()) + 1):
            model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(values)
            labels = model.predict(sample)
            sizes = pd.Series(model.labels_).value_counts(normalize=True)
            silhouette = (
                silhouette_score(sample, labels)
                if 1 < len(set(labels)) < len(sample)
                else float("nan")
            )
            records.append(
                {
                    "component": component,
                    "k": k,
                    "inertia": model.inertia_,
                    "silhouette": silhouette,
                    "smallest_cluster_share": sizes.min(),
                }
            )
    return pd.DataFrame(records)


def recommend_clusters(diagnostics: pd.DataFrame) -> dict[str, int]:
    """Reproducible elbow heuristic, not proof of an optimal segmentation.

    Choose maximum vertical distance below the normalized endpoint chord. Prefer
    candidates with at least 1% of train snapshots in each cluster. Review plots.
    """
    selected = {}
    for component, rows in diagnostics.groupby("component", sort=False):
        rows = rows.sort_values("k").copy()
        if len(rows) < 3:
            raise ValueError("Need at least three candidate k values for elbow diagnostics")
        x = (rows.k - rows.k.min()) / (rows.k.max() - rows.k.min())
        y = (rows.inertia - rows.inertia.iloc[-1]) / (rows.inertia.iloc[0] - rows.inertia.iloc[-1])
        rows["elbow_distance"] = (1 - x) - y
        candidates = rows.loc[
            (rows.k > 1) & (rows.k < rows.k.max()) & rows.smallest_cluster_share.ge(0.01)
        ]
        if candidates.empty:
            raise ValueError(f"No usable elbow candidate for {component}; review manually")
        selected[component] = int(candidates.loc[candidates.elbow_distance.idxmax(), "k"])
    return selected


def plot_cluster_diagnostics(diagnostics: pd.DataFrame, selected: dict[str, int]):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), layout="constrained")
    for row, (component, values) in zip(
        axes, diagnostics.groupby("component", sort=False), strict=True
    ):
        for ax, metric, title in zip(
            row,
            ["inertia", "silhouette", "smallest_cluster_share"],
            ["Inertia / elbow", "Sample silhouette", "Smallest cluster share"],
            strict=True,
        ):
            ax.plot(values.k, values[metric], marker="o")
            ax.axvline(selected[component], color="tab:orange", linestyle="--")
            ax.set(
                title=f"{component.upper()}: {title}", xlabel="Number of clusters", xticks=values.k
            )
            ax.grid(alpha=0.2)
    return fig, axes


def segment_profiles(scored: pd.DataFrame) -> pd.DataFrame:
    """One selected cutoff only: each customer contributes once."""
    if scored.cutoff.nunique() != 1 or scored["Customer ID"].duplicated().any():
        raise ValueError("Profiles require one cutoff and one row per customer")
    return scored.groupby("rfm_segment").agg(
        customers=("Customer ID", "size"),
        recency_median=("recency_days", "median"),
        invoices_median=("invoice_count", "median"),
        spend_median=("monetary_total", "median"),
        score_median=("rfm_score", "median"),
    )


def plot_customer_groups(scored: pd.DataFrame):
    """Full selected-cutoff population; no validation/test data or labels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for segment, group in scored.groupby("rfm_segment"):
        axes[0].scatter(group.recency_days, group.invoice_count, s=10, alpha=0.35, label=segment)
        axes[1].scatter(group.invoice_count, group.monetary_total, s=10, alpha=0.35, label=segment)
    axes[0].set(xlabel="Recency (days)", ylabel="Invoices", yscale="log")
    axes[1].set(xlabel="Invoices", ylabel="Recorded spend (£)", xscale="log", yscale="log")
    counts = scored.rfm_segment.value_counts().reindex(["low", "medium", "high"], fill_value=0)
    axes[2].bar(counts.index, counts)
    axes[2].set(xlabel="Relative RFM segment", ylabel="Customers")
    axes[0].legend(title="Segment")
    fig.suptitle(f"Customer groups at training cutoff {scored.cutoff.iloc[0].date()}")
    return fig, axes
