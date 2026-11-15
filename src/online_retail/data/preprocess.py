"""Build identified, positive-price purchase lines and an auditable cleansing report.

Run from the repository root:
    uv run python -m online_retail.data.preprocess
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from online_retail.data.load_data import COLUMNS, STRING_COLUMNS, load_data

# Deliberately explicit: do not infer validity from stock-code length.
EXCLUDED_STOCK_CODES = {"TEST001", "TEST002", "ADJUST2", "BANK CHARGES"}


def clean_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Filter purchase lines and sum quantities when all other fields match.

    Rules are sequential: removal counts sum to rows excluded before aggregation.
    Missing descriptions and country values are reported, not imputed.
    """
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = frame[COLUMNS].copy()
    for column in STRING_COLUMNS:
        data[column] = data[column].astype("string")
    for column in ("Invoice", "StockCode", "Customer ID"):
        data[column] = data[column].astype("string").str.strip().replace("", pd.NA)
    for column in ("Quantity", "Price"):
        # Fail on malformed values rather than silently treating them as purchases.
        data[column] = pd.to_numeric(data[column], errors="raise")
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="raise")

    report = {
        "input_rows": len(data),
        "missing_before": {c: int(v) for c, v in data.isna().sum().items()},
        "exact_duplicate_rows_before": int(data.duplicated().sum()),
        "excluded_stock_codes": sorted(EXCLUDED_STOCK_CODES),
        "removed_by_rule": {},
    }
    rules = {
        "missing_customer_id": data["Customer ID"].isna(),
        "missing_required_purchase_fields": data[
            ["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate"]
        ]
        .isna()
        .any(axis=1),
        "nonfinite_quantity_or_price": data[["Quantity", "Price"]]
        .isin([float("inf"), float("-inf")])
        .any(axis=1),
        "cancellation_invoice": data["Invoice"].str.upper().str.startswith("C", na=False),
        "nonpositive_quantity": data["Quantity"].le(0),
        "nonpositive_price": data["Price"].le(0),
        "excluded_stock_code": data["StockCode"].str.upper().isin(EXCLUDED_STOCK_CODES),
    }
    keep = pd.Series(True, index=data.index)
    for name, mask in rules.items():
        removed = keep & mask.fillna(False)
        report["removed_by_rule"][name] = int(removed.sum())
        keep &= ~mask.fillna(False)
    cleaned = data.loc[keep].copy().reset_index(drop=True)
    # An invoice must not silently become multiple customers' orders downstream.
    customer_counts = cleaned.groupby("Invoice")["Customer ID"].nunique()
    conflicts = customer_counts[customer_counts > 1]
    if not conflicts.empty:
        raise ValueError(
            f"Invoices associated with multiple customers: {conflicts.index.tolist()[:10]}"
        )
    # Reject fractional counts rather than silently rounding or truncating them.
    if cleaned["Quantity"].mod(1).ne(0).any():
        raise ValueError("Quantity must contain whole integers")
    cleaned["Quantity"] = cleaned["Quantity"].astype("int64")
    cleaned["Price"] = cleaned["Price"].astype("float64")
    report["rows_before_aggregation"] = len(cleaned)
    # Price and invoice metadata remain keys: no arbitrary price or description
    # is selected when lines disagree. dropna=False preserves missing descriptions.
    group_columns = [column for column in COLUMNS if column != "Quantity"]
    cleaned = cleaned.groupby(group_columns, dropna=False, sort=False, as_index=False)[
        "Quantity"
    ].sum()[COLUMNS]
    report["rows_consolidated"] = report["rows_before_aggregation"] - len(cleaned)
    report["aggregation_keys"] = group_columns
    report.update(
        {
            "output_rows": len(cleaned),
            "unique_customers": int(cleaned["Customer ID"].nunique()),
            "unique_invoices": int(cleaned["Invoice"].nunique()),
            "missing_after": {c: int(v) for c, v in cleaned.isna().sum().items()},
            "exact_duplicate_rows_after": int(cleaned.duplicated().sum()),
        }
    )
    return cleaned, report


def main() -> None:
    """Save purchase lines to Parquet and a source-linked JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/online_retail_II.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/purchases.parquet"))
    args = parser.parse_args()
    cleaned, report = clean_data(load_data(args.input))
    report["source"] = str(args.input)
    with args.input.open("rb") as source:
        report["source_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(args.output, index=False)
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved {args.output} and {report_path}")


if __name__ == "__main__":
    main()
