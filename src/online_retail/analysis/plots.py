"""Simple plots for cleaned purchase lines (positive sales, not net revenue)."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

METRICS = {
    "revenue": "Recorded sales revenue (£)",
    "quantity": "Units sold",
    "transactions": "Transactions (distinct invoices)",
    "customers": "Active customers",
}


def sales_over_time(data: pd.DataFrame, frequency: str = "MS") -> pd.DataFrame:
    """Aggregate line values; count invoices and customers once per time bucket.

    Missing buckets are filled with zero, assuming continuous source coverage.
    Customer counts are unique within each bucket, not additive across buckets.
    """
    lines = data.assign(revenue=data["Quantity"] * data["Price"])
    return (
        lines.set_index("InvoiceDate")
        .resample(frequency)
        .agg(
            revenue=("revenue", "sum"),
            quantity=("Quantity", "sum"),
            transactions=("Invoice", "nunique"),
            customers=("Customer ID", "nunique"),
        )
        .fillna(0)
    )


def plot_sales_over_time(summary: pd.DataFrame, title: str = "Monthly purchase activity"):
    """Plot comparable time axes without mixing metrics on a shared y-axis."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, layout="constrained")
    for ax, (metric, label) in zip(axes, METRICS.items(), strict=True):
        ax.plot(summary.index, summary[metric], marker="o", markersize=3)
        ax.set_ylabel(label)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(bottom=0)
    locator = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.suptitle(title)
    return fig, axes


def plot_monthly_seasonality(monthly: pd.DataFrame):
    """Compare calendar months by year; pass only complete months."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout="constrained")
    for ax, (metric, label) in zip(axes.flat, METRICS.items(), strict=True):
        for year, values in monthly.groupby(monthly.index.year):
            ax.plot(values.index.month, values[metric], marker="o", label=str(year))
        ax.set(title=label, xticks=range(1, 13), xlabel="Calendar month", ylim=(0, None))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(alpha=0.25)
        ax.legend(title="Year")
    fig.suptitle("Calendar-month comparison — boundary months excluded")
    return fig, axes


def transactions_by_country(data: pd.DataFrame) -> pd.DataFrame:
    """Count distinct invoices by country, with shares of all retained invoices.

    Fail if an invoice maps to multiple countries rather than double-counting it.
    """
    if data.groupby("Invoice")["Country"].nunique(dropna=False).gt(1).any():
        raise ValueError("An invoice is associated with multiple countries")
    orders = data[["Invoice", "Country"]].drop_duplicates("Invoice")
    countries = (
        orders.groupby("Country", dropna=False)
        .size()
        .rename("transactions")
        .sort_values(ascending=False)
        .to_frame()
    )
    countries["share_pct"] = countries["transactions"] / len(orders) * 100
    return countries


def plot_country_transactions(countries: pd.DataFrame, top_n: int = 15):
    """Show all-country and non-UK rankings so UK dominance does not hide detail.

    Shares remain relative to all invoices in both panels. Omitted countries
    remain in the input table; this does not create a modeling 'Other' category.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    panels = [
        (countries.head(top_n), f"Top {top_n} countries"),
        (countries.drop(index="United Kingdom", errors="ignore").head(top_n), "Outside the UK"),
    ]
    for ax, (values, title) in zip(axes, panels, strict=True):
        values = values.iloc[::-1]
        labels = values.index.to_series().fillna("Missing country").astype(str)
        bars = ax.barh(labels, values["transactions"])
        ax.bar_label(bars, labels=[f" {share:.2f}%" for share in values["share_pct"]], fontsize=9)
        ax.set(title=title, xlabel="Distinct invoices")
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.margins(x=0.25)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Geographical distribution of transactions — shares of all invoices")
    return fig, axes
