"""Versioned local bundle: trusted fitted pipeline, policy and integrity manifest."""

import hashlib
import importlib.metadata
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from online_retail.analysis import customers
from online_retail.features import snapshots


def checksum(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def feature_version():
    """Identity of the actual shared feature definitions used during packaging."""
    digest = hashlib.sha256()
    for module in (snapshots, customers):
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


def validate_policy(policy):
    columns = policy.get("feature_columns", [])
    if (
        not columns
        or not all(isinstance(c, str) for c in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("Policy must define unique feature columns")
    if set(columns) & {"Customer ID", "cutoff", snapshots.TARGET}:
        raise ValueError("Identifiers, cutoff and target cannot be model features")
    threshold = policy.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (float, int))
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Policy threshold must be finite and within [0, 1]")
    for field in ("fit_last_cutoff", "policy_last_cutoff"):
        date = pd.Timestamp(policy.get(field))
        if pd.isna(date) or date.tzinfo is not None:
            raise ValueError(f"Policy needs a timezone-naive {field}")


def validate_pipeline(model, policy):
    validate_policy(policy)
    if list(model.feature_names_in_) != policy["feature_columns"]:
        raise ValueError("Pipeline input order does not match policy feature schema")
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Pipeline must predict classes [0, 1] in that order")


def create_bundle(source: Path, destination: Path):
    """Copy only frozen model/policy. Never train, select or overwrite artifacts."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError("Bundle destination already exists")
    model_path, policy_path = source / "model.pkl", source / "frozen_policy.json"
    policy = json.loads(policy_path.read_text())
    with model_path.open("rb") as handle:  # Trusted local pickle only.
        model = pickle.load(handle)
    validate_pipeline(model, policy)
    available = max(pd.Timestamp(policy[k]) for k in ("fit_last_cutoff", "policy_last_cutoff"))
    manifest = {
        "bundle_schema_version": 1,
        "model_version": checksum(model_path),
        "policy_version": checksum(policy_path),
        "feature_version": feature_version(),
        "target": snapshots.TARGET,
        "horizon_days": snapshots.HORIZON.days,
        "available_from": str(available + snapshots.HORIZON),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("online-retail", "scikit-learn", "xgboost", "pandas", "numpy")
        },
        "provenance": "Hashes and versions recorded at packaging; no training performed",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        staged = Path(temp) / "bundle"
        staged.mkdir()
        (staged / "model.pkl").write_bytes(model_path.read_bytes())
        (staged / "frozen_policy.json").write_bytes(policy_path.read_bytes())
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Renaming an assembled directory avoids leaving a half-written bundle.
        staged.rename(destination)
    return manifest


def load_bundle(folder: Path):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("bundle_schema_version") != 1:
        raise ValueError("Unsupported bundle schema version")
    if manifest.get("target") != snapshots.TARGET or manifest.get("horizon_days") != 50:
        raise ValueError("Bundle prediction contract does not match this project")
    for filename, version in (
        ("model.pkl", "model_version"),
        ("frozen_policy.json", "policy_version"),
    ):
        if checksum(folder / filename) != manifest[version]:
            raise ValueError(f"Bundle checksum mismatch: {filename}")
    if manifest["feature_version"] != feature_version():
        raise ValueError("Feature definitions changed since bundle packaging")
    policy = json.loads((folder / "frozen_policy.json").read_text())
    with (folder / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    validate_pipeline(model, policy)
    expected = max(pd.Timestamp(policy[k]) for k in ("fit_last_cutoff", "policy_last_cutoff"))
    if pd.Timestamp(manifest["available_from"]) != expected + snapshots.HORIZON:
        raise ValueError("Bundle availability does not match policy label maturity")
    return model, policy, manifest
