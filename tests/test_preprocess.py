import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from online_retail.data.load_data import load_data
from online_retail.data.preprocess import clean_data


def purchase(**changes):
    return {
        "Invoice": "100",
        "StockCode": "12345A",
        "Description": None,
        "Quantity": 1,
        "InvoiceDate": pd.Timestamp("2010-01-01"),
        "Price": 2,
        "Customer ID": "42",
        "Country": "Unspecified",
        **changes,
    }


def test_purchase_rules_aggregate_duplicates_and_preserve_input():
    rows = [purchase(), purchase()]
    rows += [
        purchase(**change)
        for change in [
            {"Customer ID": None},
            {"Invoice": "C101"},
            {"Invoice": "c102"},
            {"Quantity": -1},
            {"Quantity": 0},
            {"Price": -1},
            {"Price": 0},
            {"StockCode": "TEST001"},
            {"StockCode": "TEST002"},
            {"StockCode": "ADJUST2"},
            {"StockCode": "BANK CHARGES"},
            {"InvoiceDate": None},
            {"Invoice": None},
            {"Price": float("inf")},
            {"Customer ID": None, "Invoice": "C999", "Quantity": -1},
        ]
    ]
    original = pd.DataFrame(rows)
    before = original.copy(deep=True)
    cleaned, report = clean_data(original)
    assert_frame_equal(original, before)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "Quantity"] == 2
    assert cleaned["Description"].isna().all()
    assert report["exact_duplicate_rows_after"] == 0
    assert report["rows_before_aggregation"] == 2
    assert report["rows_consolidated"] == 1
    assert sum(report["removed_by_rule"].values()) == len(original) - 2
    assert report["removed_by_rule"]["missing_customer_id"] == 2
    assert_frame_equal(clean_data(cleaned)[0], cleaned)


def test_aggregation_preserves_quantity_value_and_types(tmp_path):
    original = pd.DataFrame(
        [
            purchase(Quantity=1.0),
            purchase(Quantity=3.0),
            purchase(Price=3.0),
            purchase(Invoice="200"),
            purchase(StockCode="other"),
        ]
    )
    cleaned, _ = clean_data(original)
    assert len(cleaned) == 4
    assert cleaned["Quantity"].sum() == original["Quantity"].sum()
    assert (cleaned["Quantity"] * cleaned["Price"]).sum() == (
        original["Quantity"] * original["Price"]
    ).sum()
    for column in ("Invoice", "StockCode", "Description", "Customer ID", "Country"):
        assert isinstance(cleaned[column].dtype, pd.StringDtype)
    assert str(cleaned["Quantity"].dtype) == "int64"
    assert str(cleaned["Price"].dtype) == "float64"
    assert pd.api.types.is_datetime64_any_dtype(cleaned["InvoiceDate"])
    path = tmp_path / "purchases.parquet"
    cleaned.to_parquet(path, index=False)
    assert_frame_equal(pd.read_parquet(path), cleaned)


def test_fractional_quantity_rejected():
    with pytest.raises(ValueError, match="whole integers"):
        clean_data(pd.DataFrame([purchase(Quantity=1.5)]))


def test_customer_conflict_rejected():
    with pytest.raises(ValueError, match="multiple customers"):
        clean_data(pd.DataFrame([purchase(), purchase(**{"Customer ID": "43"})]))


def test_loader_reads_all_sheets_and_preserves_codes(tmp_path):
    path = tmp_path / "retail.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([purchase(StockCode="00123")]).to_excel(writer, sheet_name="one", index=False)
        pd.DataFrame([purchase(Invoice="200")]).to_excel(writer, sheet_name="two", index=False)
    data = load_data(path)
    assert len(data) == 2
    assert data.loc[0, "StockCode"] == "00123"
    assert data.loc[1, "Invoice"] == "200"
