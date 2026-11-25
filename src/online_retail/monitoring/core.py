"""Fixed development reference, missingness and histogram drift; no new dependencies."""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from online_retail.serving.bundle import checksum, load_bundle
from online_retail.serving.inference import predict_features

DRIFT_FEATURES = (
    "recency_days",
    "invoice_count",
    "monetary_total",
    "invoice_count_30d",
    "invoice_count_90d",
    "average_invoice_value",
    "last_gap_days",
    "observed_history_days",
)
HISTORY_GROUPS = ("1", "2-3", "4+")
DEFAULT_LIMITS = {
    "distribution_tv": 0.20,
    "missing_increase": 0.05,
    "contact_rate_change": 0.05,
    "minimum_rows": 100,
    "minimum_population_ratio": 0.50,
}
IDENTITIES = ("model_version", "policy_version", "feature_version")


def history_groups(invoice_count):
    if invoice_count.isna().any() or invoice_count.lt(1).any():
        raise ValueError("History groups require positive observed invoice counts")
    return pd.Series(
        np.select([invoice_count.eq(1), invoice_count.le(3)], ["1", "2-3"], default="4+"),
        index=invoice_count.index,
    )


def history_fingerprint(purchases, cutoff):
    """Order-independent identity of the purchase fields used by past-only features."""
    columns = ["Customer ID", "Invoice", "InvoiceDate", "StockCode", "Quantity", "Price"]
    history = purchases.loc[purchases.InvoiceDate.lt(pd.Timestamp(cutoff)), columns].copy()
    for column in ("Customer ID", "Invoice", "StockCode"):
        history[column] = history[column].astype(str)
    history["InvoiceDate"] = history.InvoiceDate.astype("datetime64[ns]")
    history["Quantity"] = history.Quantity.astype("int64")
    history["Price"] = history.Price.astype("float64")
    hashes = np.sort(pd.util.hash_pandas_object(history, index=False).to_numpy())
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def _values(series):
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("Monitoring features must be finite or missing")
    return values


def _mass(series, profile):
    values = _values(series)
    cuts = profile["cuts"]
    bins = np.searchsorted(cuts, values, side="right") + 1
    if profile["minimum"] is not None:
        bins[values < profile["minimum"]] = 0
        bins[values > profile["maximum"]] = len(cuts) + 2
    bins[np.isnan(values)] = len(cuts) + 3
    return (np.bincount(bins, minlength=len(cuts) + 4) / len(values)).tolist()


def profile(series, probability=False):
    values = _values(series)
    observed = values[~np.isnan(values)]
    if probability:
        minimum, maximum, cuts = 0.0, 1.0, np.arange(0.1, 1.0, 0.1).tolist()
    elif len(observed):
        minimum, maximum = float(observed.min()), float(observed.max())
        cuts = [
            float(v)
            for v in np.unique(np.quantile(observed, np.linspace(0, 1, 11)))
            if minimum < v < maximum
        ]
    else:
        minimum, maximum, cuts = None, None, []
    result = {
        "minimum": minimum,
        "maximum": maximum,
        "cuts": cuts,
        "missing_rate": float(np.isnan(values).mean()),
        "mean": float(observed.mean()) if len(observed) else None,
        "median": float(np.median(observed)) if len(observed) else None,
    }
    result["mass"] = _mass(series, result)
    return result


def total_variation(reference_mass, current_mass):
    return float(np.abs(np.array(reference_mass) - np.array(current_mass)).sum() / 2)


def _cohorts(groups):
    return [float(groups.eq(group).mean()) for group in HISTORY_GROUPS]


def build_reference(data_path, bundle, output, limits=None):
    data_path, bundle, output = map(Path, (data_path, bundle, output))
    if output.exists():
        raise FileExistsError("Monitoring reference already exists")
    settings = {**DEFAULT_LIMITS, **(limits or {})}
    if set(settings) != set(DEFAULT_LIMITS):
        raise ValueError("Unknown monitoring settings")
    for key, value in settings.items():
        if not np.isfinite(value) or value < 0 or (key != "minimum_rows" and value > 1):
            raise ValueError("Invalid monitoring limit")
    model, policy, manifest = load_bundle(bundle)
    columns = policy["feature_columns"]
    cutoff = pd.Timestamp(policy["policy_last_cutoff"])
    if cutoff >= pd.Timestamp(manifest["available_from"]):
        raise ValueError("Reference must precede frozen-policy availability")
    # Read features only: targets are not inputs to drift monitoring.
    data = pd.read_parquet(data_path, columns=["Customer ID", "cutoff", *columns])
    features = data.loc[data.cutoff.eq(cutoff)].reset_index(drop=True)
    if features.empty or features["Customer ID"].isna().any():
        raise ValueError("No identified reference population at the last policy cutoff")
    scores = predict_features(model, policy, features)
    reference = {
        "schema_version": 1,
        "population": "one customer population at the last policy-holdout cutoff",
        "cutoff": str(cutoff),
        "rows": len(features),
        "source_sha256": checksum(data_path),
        **{key: manifest[key] for key in IDENTITIES},
        "feature_columns": columns,
        "threshold": policy["threshold"],
        "limits": settings,
        "features": {name: profile(features[name]) for name in columns},
        "probabilities": profile(scores.purchase_probability, probability=True),
        "contact_rate": float(scores.contact.mean()),
        "history_group_mass": _cohorts(history_groups(features.invoice_count)),
        "drift_features": [name for name in DRIFT_FEATURES if name in columns],
        "performance_reference": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "reference"
        stage.mkdir()
        (stage / "reference.json").write_text(
            json.dumps(reference, indent=2, allow_nan=False) + "\n"
        )
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return reference


def monitor_batch(features, scores, reference_path, manifest, threshold):
    reference_path = Path(reference_path)
    reference = json.loads(reference_path.read_text())
    if reference.get("schema_version") != 1:
        raise ValueError("Unsupported monitoring reference schema")
    if (
        any(reference[key] != manifest[key] for key in IDENTITIES)
        or reference["threshold"] != threshold
    ):
        raise ValueError(
            "Monitoring reference belongs to a different model, policy or feature version"
        )
    if list(reference["feature_columns"]) != list(reference["features"]):
        raise ValueError("Monitoring reference schema is inconsistent")
    if features.empty or len(features) != len(scores):
        raise ValueError("Monitoring requires aligned nonempty features and scores")
    if not features[["Customer ID", "cutoff"]].equals(scores[["Customer ID", "cutoff"]]):
        raise ValueError("Monitoring feature/score keys do not align")
    limits, warnings, columns = reference["limits"], [], {}
    for name in reference["feature_columns"]:
        current = profile(features[name])
        baseline = reference["features"][name]
        distance = total_variation(baseline["mass"], _mass(features[name], baseline))
        missing_change = current["missing_rate"] - baseline["missing_rate"]
        columns[name] = {
            "reference_median": baseline["median"],
            "current_median": current["median"],
            "reference_missing_rate": baseline["missing_rate"],
            "current_missing_rate": current["missing_rate"],
            "missing_increase": missing_change,
            "distribution_tv": distance if name in reference["drift_features"] else None,
        }
        if missing_change > limits["missing_increase"]:
            warnings.append(f"missingness:{name}")
        if name in reference["drift_features"] and distance > limits["distribution_tv"]:
            warnings.append(f"feature_distribution:{name}")
    current_mass = _mass(scores.purchase_probability, reference["probabilities"])
    probability_tv = total_variation(reference["probabilities"]["mass"], current_mass)
    cohort_mass = _cohorts(history_groups(features.invoice_count))
    cohort_tv = total_variation(reference["history_group_mass"], cohort_mass)
    contact_rate = float(scores.contact.mean())
    if probability_tv > limits["distribution_tv"]:
        warnings.append("prediction_distribution")
    if cohort_tv > limits["distribution_tv"]:
        warnings.append("history_group_mix")
    if abs(contact_rate - reference["contact_rate"]) > limits["contact_rate_change"]:
        warnings.append("contact_rate")
    if (
        len(features) < limits["minimum_rows"]
        or len(features) / reference["rows"] < limits["minimum_population_ratio"]
    ):
        warnings.append("small_population")
    return {
        "schema_version": 1,
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "cutoff": str(features.cutoff.iloc[0]),
        "rows": len(features),
        "reference_cutoff": reference["cutoff"],
        "reference_rows": reference["rows"],
        "reference_version": checksum(reference_path),
        **{key: manifest[key] for key in IDENTITIES},
        "limits": limits,
        "features": columns,
        "probability_distribution_tv": probability_tv,
        "reference_probability_mass": reference["probabilities"]["mass"],
        "current_probability_mass": current_mass,
        "reference_contact_rate": reference["contact_rate"],
        "current_contact_rate": contact_rate,
        "history_group_labels": list(HISTORY_GROUPS),
        "reference_history_group_mass": reference["history_group_mass"],
        "current_history_group_mass": cohort_mass,
        "history_group_distribution_tv": cohort_tv,
        "labels_used": False,
        "performance_degradation_established": False,
    }
