"""Training-fitted, ordered RFM clusters. No classifier is trained here."""

import math

import pandas as pd
from sklearn.cluster import KMeans

RFM = {"r": "recency_days", "f": "invoice_count", "m": "monetary_total"}


class RFMSegmenter:
    """Three clusters per component; pooled training snapshots are equally weighted."""

    def fit(self, features: pd.DataFrame):
        self.models_ = {}
        self.scores_ = {}
        for component, column in RFM.items():
            values = features[[column]].copy()
            if component != "r":
                values[column] = values[column].map(math.log1p)
            if values[column].nunique() < 3:
                raise ValueError(f"Need three distinct training values for {column}")
            model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(values)
            ordered = sorted(
                range(3), key=lambda i: model.cluster_centers_[i, 0], reverse=component == "r"
            )
            self.models_[component] = model
            self.scores_[component] = {cluster: score for score, cluster in enumerate(ordered, 1)}
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        result = features.copy()
        for component, column in RFM.items():
            values = features[[column]].copy()
            if component != "r":
                values[column] = values[column].map(math.log1p)
            clusters = self.models_[component].predict(values)
            result[f"{component}_score"] = (
                pd.Series(clusters, index=result.index).map(self.scores_[component]).astype("int8")
            )
        result["rfm_score"] = result[["r_score", "f_score", "m_score"]].sum(axis=1).astype("int8")
        result["rfm_segment"] = pd.cut(
            result.rfm_score, bins=[2, 4, 7, 9], labels=["low", "medium", "high"]
        ).astype("string")
        return result
