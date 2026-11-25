"""Auditable local batch scoring with aggregate structured events."""

import json
import logging
import tempfile
import time
import uuid
from pathlib import Path

import pandas as pd

from online_retail.serving.bundle import checksum, load_bundle
from online_retail.serving.inference import predict_features, scoring_features


def run_scoring(
    purchases_path: Path,
    bundle: Path,
    cutoff,
    coverage_end,
    output: Path,
    logger=None,
    run_id=None,
    monitoring_reference=None,
):
    logger = logger or logging.getLogger("online_retail")
    run_id = run_id or uuid.uuid4().hex
    started = time.perf_counter()
    context = {"run_id": run_id, "cutoff": str(cutoff)}
    logger.info("scoring_started", extra={"context": context})
    try:
        if output.exists():
            raise FileExistsError("Scoring output already exists; use a new run directory")
        model, policy, manifest = load_bundle(bundle)
        if pd.Timestamp(cutoff) < pd.Timestamp(manifest["available_from"]):
            raise ValueError("Cutoff precedes completed model/policy label windows")
        purchases = pd.read_parquet(purchases_path)
        features = scoring_features(purchases, cutoff, coverage_end)
        logger.info(
            "features_built",
            extra={
                "context": {
                    **context,
                    "input_rows": len(purchases),
                    "eligible_customers": len(features),
                }
            },
        )
        scores = predict_features(model, policy, features)
        versions = {k: manifest[k] for k in ("model_version", "policy_version", "feature_version")}
        versions["data_version"] = checksum(purchases_path)
        for key, value in versions.items():
            scores[key] = value
        scores["run_id"] = run_id
        metadata = {
            **context,
            **versions,
            "coverage_end_exclusive": str(pd.Timestamp(coverage_end)),
            "threshold": policy["threshold"],
            "prediction_horizon_days": manifest["horizon_days"],
            "input_rows": len(purchases),
            "scored_customers": len(scores),
            "contacts": int(scores.contact.sum()),
            "mean_probability": float(scores.purchase_probability.mean()),
            "feature_missing_counts": {
                k: int(v) for k, v in features[policy["feature_columns"]].isna().sum().items()
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        monitoring = None
        if monitoring_reference is not None:
            from online_retail.monitoring.core import (
                history_fingerprint,
                history_groups,
                monitor_batch,
            )

            monitoring = monitor_batch(
                features, scores, monitoring_reference, manifest, policy["threshold"]
            )
            scores["history_group"] = history_groups(features.invoice_count)
            metadata["history_sha256"] = history_fingerprint(purchases, cutoff)
            metadata["monitoring_status"] = monitoring["status"]
            metadata["monitoring_reference_version"] = monitoring["reference_version"]
            monitoring["run_id"] = run_id
            monitoring["data_version"] = versions["data_version"]
            logger.log(
                logging.WARNING if monitoring["warnings"] else logging.INFO,
                "monitoring_completed",
                extra={
                    "context": {
                        **context,
                        "status": monitoring["status"],
                        "warning_codes": monitoring["warnings"],
                    }
                },
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as temp:
            staged = Path(temp) / "scores"
            staged.mkdir()
            scores.to_parquet(staged / "scores.parquet", index=False)
            if monitoring is not None:
                from online_retail.monitoring.reports import write_monitoring

                write_monitoring(monitoring, staged)
            metadata["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            (staged / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
            staged.rename(output)
        logger.info(
            "scoring_succeeded",
            extra={
                "context": {
                    **context,
                    **versions,
                    "scored_customers": len(scores),
                    "contacts": int(scores.contact.sum()),
                    "elapsed_seconds": metadata["elapsed_seconds"],
                    "output": str(output),
                }
            },
        )
        return scores
    except Exception:
        logger.exception("scoring_failed", extra={"context": context})
        raise
