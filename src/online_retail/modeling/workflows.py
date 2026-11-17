"""Executable orchestration shared by notebooks and scripts."""

import json
from pathlib import Path

from online_retail.modeling.tracking import TrackingConfig, file_hash, log_result_tree, tracked_run


def run_training(
    kind,
    data_dir: Path,
    output_dir: Path,
    tracking: TrackingConfig,
    n_trials=15,
    objective_metric="weekly_roc_auc",
):
    """Track fresh development training; never invokes final evaluation."""
    from online_retail.modeling.baseline import run_baseline
    from online_retail.modeling.elastic_net import run_elastic_net
    from online_retail.modeling.trees import run_tree_models

    with tracked_run(tracking, kind, data_dir) as run_id:
        if kind == "baseline":
            result = run_baseline(data_dir, output_dir)
        elif kind == "elastic_net":
            result = run_elastic_net(data_dir, output_dir)
        elif kind == "trees":
            result = run_tree_models(data_dir, output_dir, n_trials, objective_metric)
        else:
            raise ValueError(f"Unknown training workflow: {kind}")
        log_result_tree(output_dir, data_dir)
        (output_dir / "mlflow_run.json").write_text(
            json.dumps(
                {"run_id": run_id, "tracking_uri": tracking.uri, "experiment": tracking.experiment},
                indent=2,
            )
        )
    return result


def import_results(
    output_dir: Path, data_dir: Path, tracking: TrackingConfig, name="saved-results"
):
    """Import historical reports/models with an explicit provenance tag; no retraining."""
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    with tracked_run(tracking, name, data_dir, mode="historical_artifact_import") as run_id:
        log_result_tree(output_dir, data_dir)
    return run_id


def run_evaluation(data_dir: Path, model_dir: Path, output_dir: Path, tracking: TrackingConfig):
    """Read-only frozen-model diagnostics, separated from training and policy selection."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from online_retail.modeling.evaluation import (
        grouped_metrics,
        load_evaluation,
        plot_groups,
        plot_overall,
        plot_weekly,
        summarize,
    )

    with tracked_run(tracking, "frozen-model-evaluation", data_dir, mode="evaluation") as run_id:
        data, frozen = load_evaluation(data_dir, model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        threshold, costs = frozen["threshold"], frozen["costs"]
        pd.Series(summarize(data, threshold, costs), name="test").to_csv(output_dir / "overall.csv")
        tables = {
            name: grouped_metrics(data, group, threshold, costs)
            for name, group in (
                ("weekly", "cutoff"),
                ("history", "history_group"),
                ("identity", "identity_group"),
            )
        }
        for name, table in tables.items():
            table.to_csv(output_dir / f"{name}_metrics.csv")
        curves, confusion = plot_overall(data, threshold)
        figures = {
            "curves": curves,
            "confusion": confusion,
            "weekly": plot_weekly(tables["weekly"])[0],
            "history": plot_groups(tables["history"], "Invoice history")[0],
            "identity": plot_groups(tables["identity"], "Fit membership")[0],
        }
        for name, figure in figures.items():
            figure.savefig(output_dir / f"{name}.png", dpi=120)
            plt.close(figure)
        import mlflow

        mlflow.log_dict(frozen, "policy/frozen_policy.json")
        mlflow.set_tag("evaluation.test_status", "descriptive_reuse_no_selection")
        mlflow.set_tag("evaluation.source_model_sha256", file_hash(model_dir / "model.pkl"))
        log_result_tree(output_dir, data_dir)
        (output_dir / "mlflow_run.json").write_text(json.dumps({"run_id": run_id}, indent=2))
    return tables
