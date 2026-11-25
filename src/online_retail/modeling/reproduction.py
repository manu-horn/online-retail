"""Reproduce a frozen fit without search, policy selection or test evaluation."""

import hashlib
import importlib.metadata
import json
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from online_retail.modeling.baseline import fit_baseline
from online_retail.serving.bundle import checksum, validate_pipeline

TARGET = "RepurchaseWithin50Days"
KEYS = ["Customer ID", "cutoff"]
TOLERANCE = 1e-7


def fitting_population(refit, frozen):
    """Reapply the original mature-label partition, preserving source row order."""
    if refit[KEYS].isna().any().any() or refit.duplicated(KEYS).any():
        raise ValueError("Snapshot keys must be present and unique")
    fit_end = pd.Timestamp(frozen["fit_last_cutoff"])
    policy_start = pd.Timestamp(frozen["policy_first_cutoff"])
    policy_end = pd.Timestamp(frozen["policy_last_cutoff"])
    if fit_end + pd.Timedelta(days=50) > policy_start:
        raise ValueError("Fitting labels must mature before the policy holdout")
    fit = refit.loc[refit.cutoff <= fit_end].copy()
    policy = refit.loc[refit.cutoff.between(policy_start, policy_end)].copy()
    if (
        len(fit) != frozen["fit_rows"]
        or len(policy) != frozen["policy_rows"]
        or fit.cutoff.max() != fit_end
        or policy.cutoff.min() != policy_start
        or policy.cutoff.max() != policy_end
    ):
        raise ValueError("Snapshot population does not match the frozen fit/policy boundaries")
    if set(fit[TARGET].dropna().unique()) != {0, 1} or fit[TARGET].isna().any():
        raise ValueError("Fitting labels must contain both binary classes without missing values")
    return fit, policy


def _settings(estimator):
    # Missing-value sentinels are recorded as strings for strict JSON reports.
    return {
        key: str(value) if isinstance(value, float) and not np.isfinite(value) else value
        for key, value in estimator.get_params().items()
    }


def _fingerprint(frame):
    digest = hashlib.sha256()
    digest.update(str(list(zip(frame.columns, frame.dtypes.astype(str), strict=True))).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def reproduce_fit(data_dir: Path, reference_dir: Path, output: Path):
    """Save a new pipeline only after it matches the unchanged reference holdout."""
    data_dir, reference_dir, output = map(Path, (data_dir, reference_dir, output))
    if output.exists():
        raise FileExistsError("Use a new reproduction output directory")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest.get("target") != TARGET or manifest.get("horizon_days") != 50:
        raise ValueError("Expected binary 50-day processed datasets")
    frozen = json.loads((reference_dir / "frozen_policy.json").read_text())
    if frozen["model"] != "xgboost":
        raise ValueError("This reproduction workflow expects the selected XGBoost model")
    columns = frozen["feature_columns"]
    refit = pd.read_parquet(data_dir / "refit.parquet")
    fit, policy = fitting_population(refit, frozen)
    with (reference_dir / "model.pkl").open("rb") as handle:
        original = pickle.load(handle)  # Trusted local artifact only.
    validate_pipeline(original, frozen)
    effective = original.named_steps["model"].get_params()
    if any(effective.get(k) != v for k, v in frozen["params"].items()):
        raise ValueError("Saved parameters disagree with the reference pipeline")
    # clone copies constructor settings, never learned trees or imputation statistics.
    reproduced = fit_baseline(clone(original), fit[columns], fit[TARGET])
    validate_pipeline(reproduced, frozen)
    saved = pd.read_parquet(reference_dir / "policy_predictions.parquet")
    if saved.duplicated(KEYS).any() or saved[KEYS].isna().any().any():
        raise ValueError("Reference prediction keys must be present and unique")
    if set(map(tuple, saved[KEYS].to_numpy())) != set(map(tuple, policy[KEYS].to_numpy())):
        raise ValueError("Reference policy prediction keys do not match the holdout")
    saved = saved.set_index(KEYS).loc[pd.MultiIndex.from_frame(policy[KEYS])]
    expected = saved[["p_class_0", "p_class_1"]].to_numpy()
    original_p = original.predict_proba(policy[columns])
    actual = reproduced.predict_proba(policy[columns])
    reference_matches = bool(np.allclose(original_p, expected, atol=TOLERANCE, rtol=0))
    reproduction_matches = bool(np.allclose(actual, expected, atol=TOLERANCE, rtol=0))
    threshold = frozen["threshold"]
    decisions_match = bool(np.array_equal(actual[:, 1] >= threshold, expected[:, 1] >= threshold))
    if not (reference_matches and reproduction_matches and decisions_match):
        raise ValueError(
            "Reproduction parity failed; no model published. "
            f"Reference matches={reference_matches}, reproduced matches={reproduction_matches}, "
            f"decisions match={decisions_match}, max difference={np.max(np.abs(actual - expected))}"
        )
    report = {
        "workflow": "fixed_parameter_reproduction",
        "model": frozen["model"],
        "feature_columns": columns,
        "params": frozen["params"],
        "effective_model_parameters": _settings(original.named_steps["model"]),
        "imputer_parameters": _settings(original.named_steps["impute"]),
        "threshold": threshold,
        "fit_rows": len(fit),
        "fit_first_cutoff": str(fit.cutoff.min()),
        "fit_last_cutoff": str(fit.cutoff.max()),
        "policy_rows": len(policy),
        "policy_first_cutoff": str(policy.cutoff.min()),
        "policy_last_cutoff": str(policy.cutoff.max()),
        "fit_input_fingerprint": _fingerprint(fit[KEYS + columns + [TARGET]]),
        "source_checksums": {
            "refit": checksum(data_dir / "refit.parquet"),
            "dataset_manifest": checksum(data_dir / "manifest.json"),
            "reference_model": checksum(reference_dir / "model.pkl"),
            "reference_policy": checksum(reference_dir / "frozen_policy.json"),
            "reference_predictions": checksum(reference_dir / "policy_predictions.parquet"),
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("scikit-learn", "xgboost", "pandas", "numpy")
        },
        "verification": {
            "population": "policy holdout; no test data loaded",
            "absolute_tolerance": TOLERANCE,
            "max_probability_difference": float(np.max(np.abs(actual - expected))),
            "probabilities_match": reproduction_matches,
            "reference_probabilities_match": reference_matches,
            "decisions_match": decisions_match,
        },
        "hyperparameter_search": False,
        "policy_reselected": False,
        "test_evaluated": False,
    }
    predictions = policy[KEYS].copy()
    predictions[["p_class_0", "p_class_1"]] = actual
    predictions["contact"] = actual[:, 1] >= threshold
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".reproduce-") as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        (stage / "model.pkl").write_bytes(pickle.dumps(reproduced))
        shutil.copyfile(reference_dir / "frozen_policy.json", stage / "frozen_policy.json")
        predictions.to_parquet(stage / "policy_predictions.parquet", index=False)
        (stage / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return report
