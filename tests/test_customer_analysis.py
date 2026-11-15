import pandas as pd

from online_retail.analysis.customers import (
    cohort_repeat_rates,
    customer_history,
    monthly_customer_activity,
)


def test_past_only_history_invoice_counts_and_censored_cohorts():
    data = pd.DataFrame(
        {
            "Customer ID": ["A", "A", "A", "B", "C"],
            "Invoice": ["1", "1", "2", "3", "4"],
            "InvoiceDate": pd.to_datetime(
                ["2011-01-01", "2011-01-01", "2011-01-11", "2011-01-02", "2011-03-01"]
            ),
            "Quantity": [1] * 5,
            "Price": [2.0] * 5,
        }
    )
    customers, days = customer_history(data, "2011-03-01")
    assert set(customers.index) == {"A", "B"}
    assert customers.loc["A", "invoices"] == 2
    assert customers.loc["A", "monetary"] == 6
    assert days.gap_days.dropna().tolist() == [10]
    assert pd.isna(customers.loc["B", "gap_mean"])
    rates = cohort_repeat_rates(customers, days, "2011-03-01").set_index("horizon_days")
    assert rates.loc[21, "repeat_rate"] == 0.5
    assert rates.loc[50, "repeat_rate"] == 0.5
    assert pd.isna(rates.loc[90, "repeat_rate"])
    activity = monthly_customer_activity(days)
    assert activity.iloc[0].active == 2
    assert activity.iloc[0].first_observed == 2
    assert activity.iloc[0].returning == 0
