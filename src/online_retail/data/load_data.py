"""Read every worksheet without dropping or aggregating invoice lines."""

from pathlib import Path

import pandas as pd

COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]
STRING_COLUMNS = ["Invoice", "StockCode", "Description", "Customer ID", "Country"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the original workbook, preserving identifiers as strings."""
    sheets = pd.read_excel(
        path,
        sheet_name=None,
        dtype=dict.fromkeys(STRING_COLUMNS, "string"),
    )
    for name, frame in sheets.items():
        if set(frame.columns) != set(COLUMNS):
            raise ValueError(f"Unexpected columns in worksheet {name!r}: {list(frame.columns)}")
    return pd.concat([frame[COLUMNS] for frame in sheets.values()], ignore_index=True)
