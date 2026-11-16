"""Build reproducible feature datasets; no classifier or feature analysis."""

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from online_retail.features.rfm import RFMSegmenter
from online_retail.features.snapshots import (
    HORIZON,
    KEYS,
    TARGET,
    build_snapshots,
    partition_snapshots,
)


def build_feature_datasets(
    source: Path, output: Path, settings: dict, cluster_counts: dict[str, int]
) -> dict:
    """Create separate validation and final-refit RFM artifacts and datasets."""
    data = pd.read_parquet(source)
    end = data.InvoiceDate.max().normalize()
    snapshots = build_snapshots(data, observation_end=end, start_date=settings["start_date"])
    parts = partition_snapshots(
        snapshots, **{k: v for k, v in settings.items() if k != "start_date"}
    )
    if any(part.empty for part in parts.values()):
        raise ValueError("One or more partitions are empty; check coverage and configured dates")
    initial = RFMSegmenter(cluster_counts).fit(parts["train"])
    final = RFMSegmenter(cluster_counts).fit(parts["refit"])
    for name in parts:
        segmenter = initial if name in ("train", "validation") else final
        parts[name] = segmenter.transform(parts[name])
    output.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_parquet(output / f"{name}.parquet", index=False)
    for name, segmenter in (("train_rfm", initial), ("refit_rfm", final)):
        with (output / f"{name}.pkl").open("wb") as handle:
            pickle.dump(segmenter, handle)
    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {
        "source": str(source),
        "source_sha256": source_hash,
        "observation_end_exclusive": str(end),
        "settings": settings,
        "horizon_days": HORIZON.days,
        "rfm_cluster_counts": cluster_counts,
        "rfm_scoring": "equal-weight mean of component ranks normalized to 0–100",
        "rfm_segment_edges": {"train": initial.segment_edges_, "refit": final.segment_edges_},
        "feature_columns": [c for c in parts["train"] if c not in [*KEYS, TARGET]],
        "keys": KEYS,
        "target": TARGET,
        "partitions": {
            name: {
                "rows": len(part),
                "customers": part["Customer ID"].nunique(),
                "first_cutoff": str(part.cutoff.min()),
                "last_cutoff": str(part.cutoff.max()),
            }
            for name, part in parts.items()
        },
        "rfm_centers_transformed": {
            name: {
                key: model.cluster_centers_.ravel().tolist()
                for key, model in segmenter.models_.items()
            }
            for name, segmenter in (("train", initial), ("refit", final))
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
