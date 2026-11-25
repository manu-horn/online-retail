"""Past-only feature construction and frozen-pipeline inference."""

import numpy as np
import pandas as pd

from online_retail.data.preprocess import EXCLUDED_STOCK_CODES
from online_retail.features.snapshots import customer_features

REQUIRED = ["Customer ID", "Invoice", "InvoiceDate", "StockCode", "Quantity", "Price"]


def scoring_features(purchases, cutoff, coverage_end):
    """coverage_end is an explicit exclusive ingestion boundary, not last sale time."""
    cutoff, coverage_end = pd.Timestamp(cutoff), pd.Timestamp(coverage_end)
    if any(pd.isna(t) or t.tzinfo is not None for t in (cutoff, coverage_end)):
        raise ValueError("Cutoff and coverage end must be timezone-naive timestamps")
    if cutoff.dayofweek != 0 or cutoff != cutoff.normalize():
        raise ValueError("Scoring cutoff must be Monday at midnight")
    if cutoff > coverage_end:
        raise ValueError("Cutoff exceeds declared source coverage")
    if set(REQUIRED) - set(purchases.columns):
        raise ValueError("Missing required cleaned purchase columns")
    if not pd.api.types.is_datetime64_any_dtype(purchases.InvoiceDate):
        raise ValueError("InvoiceDate must have datetime dtype")
    if purchases.InvoiceDate.dt.tz is not None or purchases.InvoiceDate.isna().any():
        raise ValueError("InvoiceDate must be nonmissing and timezone-naive")
    history = purchases.loc[purchases.InvoiceDate.lt(cutoff)].copy()
    if history.empty:
        raise ValueError("No identified purchase history before cutoff")
    if history[REQUIRED].isna().any().any():
        raise ValueError("Missing essential fields in cleaned purchase history")
    for name in ("Customer ID", "Invoice", "StockCode"):
        if not pd.api.types.is_string_dtype(history[name]):
            raise ValueError(f"{name} must contain string identifiers")
        if history[name].str.strip().eq("").any():
            raise ValueError(f"{name} contains empty identifiers")
    for name in ("Quantity", "Price"):
        if not pd.api.types.is_numeric_dtype(history[name]) or not np.isfinite(history[name]).all():
            raise ValueError(f"{name} must be finite numeric values")
        if history[name].le(0).any():
            raise ValueError(f"{name} must be positive purchases")
    if history.Quantity.mod(1).ne(0).any():
        raise ValueError("Quantity must contain whole counts")
    if history.Invoice.str.upper().str.startswith("C").any():
        raise ValueError("Cancellation invoices must be cleaned before scoring")
    if history.StockCode.str.upper().isin(EXCLUDED_STOCK_CODES).any():
        raise ValueError("Adjustment/test stock codes must be cleaned before scoring")
    if history.groupby("Invoice")["Customer ID"].nunique().gt(1).any():
        raise ValueError("Invoice assigned to multiple customers")
    return customer_features(history, cutoff).sort_values("Customer ID").reset_index(drop=True)


def predict_features(model, policy, features):
    """Return probability and policy decisions; no targets or future labels needed."""
    if features.empty or features.duplicated(["Customer ID", "cutoff"]).any():
        raise ValueError("Feature rows must have unique customer/cutoff keys")
    columns = policy["feature_columns"]
    if set(columns) - set(features.columns):
        raise ValueError("Model feature columns missing from scoring table")
    if features[columns].isin([np.inf, -np.inf]).any().any():
        raise ValueError("Features contain nonfinite values")
    # Missing purchase gaps remain missing for the saved imputer to handle.
    probabilities = model.predict_proba(features[columns])
    if probabilities.shape != (len(features), 2) or not np.isfinite(probabilities).all():
        raise ValueError("Model returned invalid probability shape or values")
    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError("Model probabilities must be in [0, 1] and sum to one")
    scores = features[["Customer ID", "cutoff"]].copy()
    scores["purchase_probability"] = probabilities[:, 1]
    scores["contact"] = scores.purchase_probability.ge(policy["threshold"])
    return scores
