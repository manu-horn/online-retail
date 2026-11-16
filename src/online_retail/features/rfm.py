"""Ordered RFM scores with explicit cluster counts and train-fitted segment bands."""

import math

import pandas as pd
from sklearn.cluster import KMeans

RFM = {"r": "recency_days", "f": "invoice_count", "m": "monetary_total"}


def component_values(features: pd.DataFrame, component: str) -> pd.DataFrame:
    values = features[[RFM[component]]].astype(float).copy()
    if component != "r":
        values.iloc[:, 0] = values.iloc[:, 0].astype(float).map(math.log1p)
    return values.astype(float)


class RFMSegmenter:
    """Explicit k per component; equal component weights on a 0–100 scale."""

    def __init__(self, cluster_counts: dict[str, int]):
        if set(cluster_counts) != set(RFM) or any(
            not isinstance(k, int) or k < 2 for k in cluster_counts.values()
        ):
            raise ValueError("Provide integer cluster counts >= 2 for r, f and m")
        self.cluster_counts = dict(cluster_counts)

    def fit(self, features: pd.DataFrame):
        self.models_ = {}
        self.scores_ = {}
        for component, k in self.cluster_counts.items():
            values = component_values(features, component)
            if values.iloc[:, 0].nunique() < k:
                raise ValueError(f"Insufficient distinct training values for {component}")
            model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(values)
            ordered = sorted(
                range(k), key=lambda i: model.cluster_centers_[i, 0], reverse=component == "r"
            )
            self.models_[component] = model
            self.scores_[component] = {cluster: rank for rank, cluster in enumerate(ordered)}
        scores = self._score(features).rfm_score
        self.segment_edges_ = scores.quantile([1 / 3, 2 / 3]).tolist()
        if self.segment_edges_[0] >= self.segment_edges_[1]:
            raise ValueError("Tied training tertiles cannot define three segments; review scores")
        return self

    def _score(self, features: pd.DataFrame) -> pd.DataFrame:
        result = features.copy()
        for component, model in self.models_.items():
            ranks = pd.Series(
                model.predict(component_values(features, component)), index=result.index
            )
            result[f"{component}_score"] = (
                ranks.map(self.scores_[component]) / (self.cluster_counts[component] - 1) * 100
            )
        result["rfm_score"] = result[["r_score", "f_score", "m_score"]].mean(axis=1)
        return result

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        result = self._score(features)
        low, high = self.segment_edges_
        result["rfm_segment"] = "medium"
        result.loc[result.rfm_score.le(low), "rfm_segment"] = "low"
        result.loc[result.rfm_score.gt(high), "rfm_segment"] = "high"
        result["rfm_segment"] = result.rfm_segment.astype("string")
        return result
