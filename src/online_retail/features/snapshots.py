"""Weekly customer features, complete 50-day outcomes, and chronological partitions."""

import math

import pandas as pd

from online_retail.analysis.customers import customer_history

KEYS = ["Customer ID", "cutoff"]
TARGET = "RepurchaseWithin50Days"
FIT_DATE = pd.Timestamp("2011-06-01")
VALIDATION_START = pd.Timestamp("2010-12-01")
VALIDATION_END = pd.Timestamp("2011-04-01")
HORIZON = pd.Timedelta(days=50)


def customer_features(data: pd.DataFrame, cutoff) -> pd.DataFrame:
    """All observed customers; features use strictly earlier positive purchases."""
    cutoff = pd.Timestamp(cutoff)
    history = data.loc[data.InvoiceDate.lt(cutoff)]
    base, days = customer_history(history, cutoff)
    base = base.rename(
        columns={
            "invoices": "invoice_count",
            "monetary": "monetary_total",
            "purchase_days": "purchase_day_count",
            "gap_mean": "mean_gap_days",
            "gap_std": "std_gap_days",
        }
    )
    base["average_invoice_value"] = base.monetary_total / base.invoice_count
    base["average_units_per_invoice"] = base.units / base.invoice_count
    base["distinct_product_count"] = history.groupby("Customer ID").StockCode.nunique()
    for window in (30, 90):
        recent = history.loc[history.InvoiceDate.ge(cutoff - pd.Timedelta(days=window))]
        grouped = recent.assign(value=recent.Quantity * recent.Price).groupby("Customer ID")
        base[f"invoice_count_{window}d"] = grouped.Invoice.nunique().reindex(
            base.index, fill_value=0
        )
        base[f"monetary_{window}d"] = grouped.value.sum().reindex(base.index, fill_value=0)
    dates = days.groupby("Customer ID").purchase_day
    latest = dates.max()
    for lag, name in ((1, "last_gap_days"), (2, "gap_2_span_days"), (3, "gap_3_span_days")):
        previous = days.assign(previous=dates.shift(lag)).groupby("Customer ID").tail(1)
        previous = previous.set_index("Customer ID").previous
        base[name] = (latest - previous).dt.days
    base["has_previous_purchase_day"] = base.purchase_day_count.ge(2)
    base["has_four_purchase_days"] = base.purchase_day_count.ge(4)
    angle = 2 * math.pi * (cutoff.month - 1) / 12
    base["month_sin"] = math.sin(angle)
    base["month_cos"] = math.cos(angle)
    christmas = pd.Timestamp(year=cutoff.year, month=12, day=25)
    if christmas < cutoff:
        christmas = pd.Timestamp(year=cutoff.year + 1, month=12, day=25)
    base["days_until_christmas"] = (christmas - cutoff).days
    base["cutoff"] = cutoff
    return base.drop(columns=["first_purchase", "last_purchase", "units"]).reset_index()


def build_snapshots(
    data: pd.DataFrame, observation_end=None, start_date="2010-03-01"
) -> pd.DataFrame:
    """Build Monday snapshots with fully observed labels.

    observation_end is exclusive. Default excludes the final partial calendar day.
    Source coverage must be continuous; features may start with very little history.
    """
    end = (
        pd.Timestamp(observation_end)
        if observation_end is not None
        else data.InvoiceDate.max().normalize()
    )
    if end > data.InvoiceDate.max().normalize():
        raise ValueError("Observation end exceeds conservatively observed source coverage")
    frames = []
    for cutoff in pd.date_range(start_date, end - HORIZON, freq="W-MON"):
        features = customer_features(data, cutoff)
        future = data.loc[data.InvoiceDate.ge(cutoff) & data.InvoiceDate.lt(cutoff + HORIZON)]
        next_purchase = future.groupby("Customer ID").InvoiceDate.min()
        next_dates = features["Customer ID"].map(next_purchase)
        features[TARGET] = next_dates.notna().astype("int8")
        frames.append(features)
    if not frames:
        raise ValueError("Insufficient coverage to build complete weekly snapshots")
    return pd.concat(frames, ignore_index=True)


def partition_snapshots(
    snapshots: pd.DataFrame,
    validation_start=VALIDATION_START,
    validation_end=VALIDATION_END,
    fit_date=FIT_DATE,
) -> dict[str, pd.DataFrame]:
    """Refit intentionally overlaps train/validation; it is not an evaluation split."""
    validation_start = pd.Timestamp(validation_start)
    validation_end = pd.Timestamp(validation_end)
    fit_date = pd.Timestamp(fit_date)
    if not validation_start < validation_end or validation_end + HORIZON > fit_date:
        raise ValueError("Partitions must be ordered and validation labels mature before refit")
    cutoff = snapshots.cutoff
    masks = {
        "train": (cutoff < validation_start) & (cutoff + HORIZON <= validation_start),
        "validation": (cutoff >= validation_start) & (cutoff < validation_end),
        "refit": cutoff + HORIZON <= fit_date,
        "test": cutoff >= fit_date,
    }
    return {name: snapshots.loc[mask].copy().reset_index(drop=True) for name, mask in masks.items()}
