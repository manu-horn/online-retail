"""Exploratory customer tables; no training features or labels are constructed."""

import matplotlib.pyplot as plt
import pandas as pd


def customer_history(data: pd.DataFrame, cutoff) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize purchases strictly before cutoff; return customers and purchase days.

    Dates measure observed history, not true customer tenure. Gaps use distinct
    purchase days so multiple same-day invoices do not introduce zero-day gaps.
    """
    cutoff = pd.Timestamp(cutoff)
    history = data.loc[data.InvoiceDate.lt(cutoff)].copy()
    history["revenue"] = history.Quantity * history.Price
    history["purchase_day"] = history.InvoiceDate.dt.normalize()
    customers = history.groupby("Customer ID").agg(
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        invoices=("Invoice", "nunique"),
        purchase_days=("purchase_day", "nunique"),
        monetary=("revenue", "sum"),
        units=("Quantity", "sum"),
    )
    customers["recency_days"] = (cutoff - customers.last_purchase).dt.total_seconds() / 86400
    customers["observed_history_days"] = (
        cutoff - customers.first_purchase
    ).dt.total_seconds() / 86400
    days = (
        history[["Customer ID", "purchase_day"]]
        .drop_duplicates()
        .sort_values(["Customer ID", "purchase_day"])
    )
    days["gap_days"] = days.groupby("Customer ID").purchase_day.diff().dt.days
    gaps = days.groupby("Customer ID").gap_days.agg(gap_mean="mean", gap_std="std")
    return customers.join(gaps), days


def monthly_customer_activity(days: pd.DataFrame) -> pd.DataFrame:
    """Partition monthly active customers into first-observed and returning."""
    active = days.assign(month=days.purchase_day.dt.to_period("M"))
    active = active[["Customer ID", "month"]].drop_duplicates()
    first = active.groupby("Customer ID").month.min()
    active["first_observed"] = active.month.eq(active["Customer ID"].map(first))
    summary = active.groupby("month").agg(
        active=("Customer ID", "size"), first_observed=("first_observed", "sum")
    )
    summary = summary.reindex(pd.period_range(summary.index.min(), summary.index.max(), freq="M"))
    summary = summary.fillna(0).astype(int)
    summary["returning"] = summary.active - summary.first_observed
    summary.index = summary.index.to_timestamp()
    return summary


def cohort_repeat_rates(customers: pd.DataFrame, days: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Return equal-follow-up second-purchase rates at 21/50/90 days.

    A cohort/horizon is reported only if every customer has full follow-up.
    A second purchase means a different calendar day. Terminal waiting times
    remain in the denominator, unlike a completed-gap histogram.
    """
    first_two = days.groupby("Customer ID").head(2).copy()
    first_two["number"] = first_two.groupby("Customer ID").cumcount()
    second = first_two.loc[first_two.number.eq(1)].set_index("Customer ID").purchase_day
    table = customers.copy()
    table["first_day"] = table.first_purchase.dt.normalize()
    table["second_day"] = table.index.map(second)
    table["cohort"] = table.first_day.dt.to_period("M")
    records = []
    for cohort, group in table.groupby("cohort"):
        for horizon in (21, 50, 90):
            mature = (group.first_day + pd.Timedelta(days=horizon)).le(pd.Timestamp(cutoff)).all()
            returned = group.second_day.lt(group.first_day + pd.Timedelta(days=horizon))
            records.append(
                {
                    "cohort": str(cohort),
                    "horizon_days": horizon,
                    "customers": len(group),
                    "repeat_rate": returned.mean() if mature else float("nan"),
                }
            )
    return pd.DataFrame(records)


def plot_customer_distributions(customers: pd.DataFrame, days: pd.DataFrame):
    """Show full distributions; log x-axes handle the long right tails."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    series = [
        (customers.invoices, "Invoices per customer", True),
        (customers.purchase_days, "Purchase days per customer", True),
        (customers.monetary, "Recorded purchase value (£)", True),
        (customers.recency_days, "Days since latest purchase", False),
        (customers.observed_history_days, "Days since first observed purchase", False),
        (days.gap_days.dropna(), "Completed gaps between purchase days", False),
    ]
    for ax, (values, title, log) in zip(axes.flat, series, strict=True):
        values = values.sort_values()
        ax.step(values, pd.Series(range(1, len(values) + 1)) / len(values), where="post")
        ax.set(title=title, ylabel="Fraction at or below x", xlabel="Value", ylim=(0, 1))
        if log:
            ax.set_xscale("log")
        ax.grid(alpha=0.25)
    return fig, axes


def plot_customer_activity(activity: pd.DataFrame, rates: pd.DataFrame):
    """Plot customer composition and mature-cohort repeat-purchase rates."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), layout="constrained")
    activity[["first_observed", "returning"]].plot.area(ax=axes[0])
    axes[0].set(title="Monthly active customers", ylabel="Distinct customers", xlabel="Month")
    rates.pivot(index="cohort", columns="horizon_days", values="repeat_rate").plot(
        ax=axes[1], marker="o"
    )
    axes[1].set(
        title="Second purchase on a later day — mature cohorts only",
        ylabel="Repeat-purchase fraction",
        xlabel="First-observed month",
        ylim=(0, 1),
    )
    axes[1].legend(title="Follow-up days")
    axes[1].tick_params(axis="x", rotation=45)
    return fig, axes
