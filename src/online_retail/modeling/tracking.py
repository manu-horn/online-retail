"""Explicit MLflow logging for completed artifacts; never trains or selects models."""

import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import re
import subprocess
import tempfile
import tomllib
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature


@dataclass(frozen=True)
class TrackingConfig:
    root: Path
    uri: str
    experiment: str
    artifact_dir: Path | None = None

    @classmethod
    def local(cls, root: Path, storage_dir: Path | None = None):
        root = root.resolve()
        storage = Path(storage_dir).resolve() if storage_dir is not None else root
        storage.mkdir(parents=True, exist_ok=True)
        settings = tomllib.loads((root / "configs/tracking.toml").read_text())
        return cls(
            root,
            os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{storage / 'mlflow.db'}"),
            os.environ.get("MLFLOW_EXPERIMENT_NAME", settings["experiment_name"]),
            storage / "mlartifacts",
        )


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _git(root, *args):
    try:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    except FileNotFoundError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _metrics(values, prefix="", step=0):
    for key, value in values.items():
        if isinstance(value, dict):
            _metrics(value, prefix + key + ".", step)
        elif (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        ):
            name = re.sub(r"[^\w. /-]", "_", prefix + str(key))
            mlflow.log_metric(name, float(value), step=step)


@contextmanager
def tracked_run(config: TrackingConfig, name: str, data_dir: Path, mode="training"):
    """Local SQLite by default; failed workflow executions become FAILED runs."""
    mlflow.set_tracking_uri(config.uri)
    experiment = mlflow.get_experiment_by_name(config.experiment)
    if experiment is None:
        mlflow.create_experiment(
            config.experiment,
            artifact_location=None
            if config.uri.startswith(("http://", "https://"))
            else (config.artifact_dir or config.root / "mlartifacts").as_uri(),
        )
    mlflow.set_experiment(config.experiment)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags(
            {
                "workflow.mode": mode,
                "provenance.code_context": "logging-time snapshot"
                if mode == "historical_artifact_import"
                else "execution snapshot",
                "git.commit": (
                    os.environ.get("ONLINE_RETAIL_CODE_REVISION", "unknown")
                    if _git(config.root, "rev-parse", "HEAD") == "unknown"
                    else _git(config.root, "rev-parse", "HEAD")
                ),
                "git.branch": _git(config.root, "branch", "--show-current"),
                "git.dirty": (
                    "unknown"
                    if _git(config.root, "status", "--porcelain") == "unknown"
                    else str(bool(_git(config.root, "status", "--porcelain"))).lower()
                ),
                "target": "RepurchaseWithin50Days",
                "split.strategy": "expanding_temporal_50d_gap",
            }
        )
        manifest = data_dir / "manifest.json"
        if manifest.exists():
            mlflow.log_artifact(str(manifest), "provenance")
        checksums = {p.name: file_hash(p) for p in data_dir.glob("*.parquet")}
        # Checksums only: datasets and customer-level predictions are not uploaded.
        mlflow.log_dict(checksums, "provenance/dataset_checksums.json")
        mlflow.log_dict(
            {
                package: importlib.metadata.version(package)
                for package in ("mlflow", "scikit-learn", "xgboost", "lightgbm", "pandas", "numpy")
            },
            "provenance/versions.json",
        )
        for name in (
            "uv.lock",
            "pyproject.toml",
            "configs/prediction.toml",
            "configs/tracking.toml",
        ):
            path = config.root / name
            if path.exists():
                mlflow.log_artifact(str(path), "provenance")
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "source.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for folder in ("src", "scripts"):
                    for source in (config.root / folder).rglob("*.py"):
                        bundle.write(source, source.relative_to(config.root))
            mlflow.log_artifact(str(archive), "provenance")
        yield run.info.run_id


def _log_model(path, data_dir, columns):
    if not columns:
        manifest = json.loads((data_dir / "manifest.json").read_text())
        from online_retail.modeling.baseline import DERIVED_RFM

        columns = [c for c in manifest["feature_columns"] if c not in DERIVED_RFM]
    # Trusted local pipeline. Use feature-schema example only, never IDs or outcomes.
    with path.open("rb") as handle:
        pipeline = pickle.load(handle)
    example = pd.read_parquet(data_dir / "train.parquet")[columns].head(5)
    probabilities = pipeline.predict_proba(example)
    mlflow.set_tag("source.model_sha256", file_hash(path))
    mlflow.log_dict(
        {"feature_columns": columns, "probability_columns": ["p_class_0", "p_class_1"]},
        "schema.json",
    )
    info = mlflow.sklearn.log_model(
        pipeline,
        name="model",
        signature=infer_signature(example, probabilities),
        input_example=example,
        pyfunc_predict_fn="predict_proba",
        serialization_format="cloudpickle",
        code_paths=[str(Path(__file__).resolve().parents[1])],
        pip_requirements=[
            f"{p}=={importlib.metadata.version(p)}"
            for p in (
                "scikit-learn",
                "xgboost",
                "lightgbm",
                "pandas",
                "numpy",
                "scipy",
                "threadpoolctl",
            )
        ],
    )
    mlflow.set_tag("model.uri", info.model_uri)
    return info.model_uri


def log_completed_outputs(output: Path, data_dir: Path, include_trials=True):
    """Record existing reports and pipeline in the active run; no new evaluation."""
    metadata = {}
    for name in ("run.json", "frozen_policy.json"):
        if (output / name).exists():
            metadata.update(json.loads((output / name).read_text()))
    mlflow.log_params(
        {
            k: str(v)
            for k, v in metadata.items()
            if isinstance(v, (str, int, float, bool)) or v is None
        }
    )
    mlflow.log_params(metadata.get("best_params", metadata.get("params", {})))
    for path in output.iterdir():
        if path.is_file() and path.suffix in {".json", ".csv", ".png"}:
            mlflow.log_artifact(str(path), "reports")
    if (output / "validation_metrics.csv").exists():
        for row in pd.read_csv(output / "validation_metrics.csv").to_dict("records"):
            _metrics(row, f"validation.{row.get('model', 'model')}.")
    if (output / "cv_metrics.csv").exists():
        cv = pd.read_csv(output / "cv_metrics.csv")
        if "trial" in cv and include_trials:
            trials = pd.read_csv(output / "trials.csv").set_index("number")
            for trial, group in cv.groupby("trial"):
                with mlflow.start_run(run_name=f"trial-{trial:03d}", nested=True):
                    mlflow.set_tag("workflow.mode", "completed_trial_summary")
                    mlflow.log_param("trial", int(trial))
                    params = trials.loc[trial].filter(like="params_").dropna().to_dict()
                    mlflow.log_params({k.removeprefix("params_"): v for k, v in params.items()})
                    mlflow.log_metric("cv.objective", float(trials.loc[trial, "value"]))
                    for row in group.to_dict("records"):
                        _metrics(
                            {k: v for k, v in row.items() if k not in {"trial", "fold"}},
                            "fold.",
                            int(row["fold"]),
                        )
        else:
            for row in cv.to_dict("records"):
                prefix = f"cv.{row.get('model', 'model')}."
                _metrics(
                    {k: v for k, v in row.items() if k not in {"fold", "trial"}},
                    prefix,
                    int(row["fold"]),
                )
    if (output / "overall.csv").exists():
        series = pd.read_csv(output / "overall.csv", index_col=0).iloc[:, 0]
        _metrics(series.to_dict(), "evaluation.test.")
    if (output / "report.json").exists():
        _metrics(json.loads((output / "report.json").read_text()), "reported.")
    model_path = next(
        (
            output / name
            for name in ("model.pkl", "logistic_baseline.pkl")
            if (output / name).exists()
        ),
        None,
    )
    uri = _log_model(model_path, data_dir, metadata.get("feature_columns")) if model_path else None
    return uri


def log_result_tree(output: Path, data_dir: Path):
    """Tree search directories create model children and completed-trial summaries."""
    children = [p for p in output.iterdir() if p.is_dir() and (p / "run.json").exists()]
    if children:
        for child in sorted(children):
            with mlflow.start_run(run_name=child.name, nested=True):
                mlflow.set_tag("workflow.mode", "completed_model_summary")
                log_completed_outputs(child, data_dir)
    else:
        log_completed_outputs(output, data_dir)
