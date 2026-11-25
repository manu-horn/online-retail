"""Prepare past-only JSON Lines features and validate the Batch Transform schema."""

import json
import numbers
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from online_retail.serving.bundle import checksum, load_bundle
from online_retail.serving.inference import scoring_features

METADATA = {"Customer ID", "cutoff", "feature_version", "data_version", "coverage_end"}
CONTENT_TYPE = "application/jsonlines"
MAX_BYTES = 6 * 1024 * 1024


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Dates must be strings")
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is not None:
        raise ValueError("Dates must be present and timezone-naive")
    return timestamp


def validate_records(records, policy, manifest):
    columns = policy["feature_columns"]
    if not records or any(not isinstance(row, dict) for row in records):
        raise ValueError("Expected nonempty snapshot records")
    for row in records:
        if set(row) != METADATA | set(columns):
            raise ValueError("Snapshot fields must match the exact feature/metadata schema")
        if not isinstance(row["Customer ID"], str) or not row["Customer ID"].strip():
            raise ValueError("Customer ID must be a nonempty string")
        if row["feature_version"] != manifest["feature_version"]:
            raise ValueError("Snapshot feature version does not match the bundle")
        if not isinstance(row["data_version"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", row["data_version"]
        ):
            raise ValueError("Data version must be a SHA-256 identity")
        cutoff, coverage = _timestamp(row["cutoff"]), _timestamp(row["coverage_end"])
        if cutoff.dayofweek != 0 or cutoff != cutoff.normalize():
            raise ValueError("Cutoff must be Monday at midnight")
        if cutoff < _timestamp(manifest["available_from"]) or cutoff > coverage:
            raise ValueError("Cutoff is outside model availability/source coverage")
        for column in columns:
            value = row[column]
            if value is not None and (
                not isinstance(value, numbers.Real) or not np.isfinite(value)
            ):
                raise ValueError("Features must be finite numeric values or null")
    frame = pd.DataFrame(records)
    frame["cutoff"] = pd.to_datetime(frame.cutoff)
    frame[columns] = frame[columns].astype(float)
    if frame.duplicated(["Customer ID", "cutoff"]).any():
        raise ValueError("Customer/cutoff keys must be unique within each request")
    return frame


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Repeated JSON field")
        result[key] = value
    return result


def decode_records(payload):
    try:
        lines = payload.decode("utf-8").splitlines()
        return [json.loads(line, object_pairs_hook=_object) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Malformed UTF-8 JSON Lines payload") from error


def prepare_batch(purchases_path, bundle, cutoff, coverage_end, output):
    purchases_path, bundle, output = map(Path, (purchases_path, bundle, output))
    if output.exists():
        raise FileExistsError("Batch output already exists")
    _, policy, manifest = load_bundle(bundle)
    features = scoring_features(pd.read_parquet(purchases_path), cutoff, coverage_end)
    features["feature_version"] = manifest["feature_version"]
    features["data_version"] = checksum(purchases_path)
    features["coverage_end"] = str(pd.Timestamp(coverage_end))
    payload = features.to_json(orient="records", lines=True, date_format="iso", double_precision=15)
    validate_records(decode_records(payload.encode()), policy, manifest)
    summary = {
        "rows": len(features),
        "cutoff": str(pd.Timestamp(cutoff)),
        "coverage_end_exclusive": str(pd.Timestamp(coverage_end)),
        "data_version": checksum(purchases_path),
        "feature_version": manifest["feature_version"],
        "content_type": CONTENT_TYPE,
        "split_type": "Line",
        "targets_included": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "batch"
        stage.mkdir()
        (stage / "features.jsonl").write_text(payload)
        summary["payload_sha256"] = checksum(stage / "features.jsonl")
        (stage / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return summary
