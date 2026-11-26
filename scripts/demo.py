"""Replay one historical batch from the frozen training configuration."""

import argparse
import json
import tomllib
from pathlib import Path

import pandas as pd

from online_retail.features.snapshots import HORIZON
from online_retail.logging import configure_logging
from online_retail.modeling.tracking import TrackingConfig
from online_retail.modeling.workflows import import_results, run_reproduction
from online_retail.monitoring.core import build_reference
from online_retail.monitoring.outcomes import evaluate_outcomes
from online_retail.serving.bundle import create_bundle
from online_retail.serving.workflow import run_scoring


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New demo directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / "data/processed"
    purchases = root / "data/interim/purchases.parquet"
    frozen = root / "artifacts/final_roc_policy"
    for path in (
        purchases,
        *(data / name for name in ("manifest.json", "refit.parquet", "train.parquet")),
        *(
            frozen / name
            for name in ("model.pkl", "frozen_policy.json", "policy_predictions.parquet")
        ),
    ):
        if not path.is_file():
            parser.error(f"Missing {path}. Prepare the inputs described in docs/demo.md.")

    cutoff = pd.Timestamp("2011-06-06")
    coverage_end = pd.Timestamp(
        json.loads((data / "manifest.json").read_text())["observation_end_exclusive"]
    )
    if pd.isna(coverage_end) or coverage_end.tzinfo or coverage_end < cutoff + HORIZON:
        parser.error("The historical replay requires 50 days of complete follow-up.")

    output = args.output.resolve()
    output.mkdir(parents=True)
    logger = configure_logging(output / "events.jsonl")
    tracking = TrackingConfig(
        root, f"sqlite:///{output / 'mlflow.db'}", "retail-batch-demo", output / "mlartifacts"
    )
    fit, bundle, reference = (output / name for name in ("fit", "bundle", "reference"))
    batch, outcomes = output / "batch", output / "outcomes"
    try:
        report = run_reproduction(data, frozen, fit, tracking)
        create_bundle(fit, bundle)
        limits = tomllib.loads((root / "configs/monitoring.toml").read_text())
        build_reference(data / "refit.parquet", bundle, reference, limits)
        scores = run_scoring(
            purchases,
            bundle,
            cutoff,
            coverage_end,
            batch,
            logger,
            monitoring_reference=reference / "reference.json",
        )
        evaluation = evaluate_outcomes(batch, purchases, coverage_end, outcomes)
        monitoring_run = import_results(batch, data, tracking, "batch-monitoring")
        outcomes_run = import_results(outcomes, data, tracking, "mature-outcomes")
        fit_run = json.loads((fit / "mlflow_run.json").read_text())["run_id"]
        warnings = json.loads((batch / "monitoring.json").read_text())["warnings"]
        metrics = evaluation["overall"]
        summary = (
            "# Retail repurchase demo\n\n"
            f"- Scoring cutoff **{cutoff.date()}** with outcomes through "
            f"**{(cutoff + HORIZON).date()}** (exclusive).\n"
            f"- Refit on **{report['fit_rows']:,}** snapshots. Maximum holdout probability "
            f"difference **{report['verification']['max_probability_difference']:.3g}**.\n"
            f"- Scored **{len(scores):,}** customers and flagged **{int(scores.contact.sum()):,}** "
            f"for contact at threshold **{evaluation['threshold']}**.\n"
            f"- Monitoring warnings: {', '.join(warnings) or 'none'}.\n"
            f"- Mature outcomes: ROC-AUC **{metrics['roc_auc']:.3f}**, "
            f"buyer precision **{metrics['buyer_precision']:.1%}** and "
            f"recall **{metrics['buyer_recall']:.1%}**.\n\n"
            "Open the [batch report](batch/monitoring.md), [chart](batch/monitoring.png) "
            "or [outcome report](outcomes/report.md).\n\n"
            "MLflow run IDs\n\n"
            f"- Training `{fit_run}`\n- Monitoring `{monitoring_run}`\n"
            f"- Mature outcomes `{outcomes_run}`\n"
        )
        (output / "summary.md").write_text(summary)
    except Exception:
        logger.exception("demo_failed")
        raise
    logger.info("demo_completed", extra={"context": {"output": str(output)}})
    print(f"Demo summary: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
