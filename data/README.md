# Data

Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

Citation: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5CG6D.
The dataset is distributed under CC BY 4.0; preserve attribution in derived outputs.

Place the original `online_retail_II.xlsx` in `data/raw/`. It is already present
in this workspace. Keep it immutable and out of Git. Later ingestion will record
its checksum and sheet names, then write normalized Parquet to `data/interim/`
and customer snapshots to `data/processed/`.

Inspect both workbook sheets before combining: check actual column names, date
ranges, overlap, missing customer IDs, cancellations, returns, and duplicates.
Do not assume the website's column names match the workbook exactly. Do not
drop identical-looking invoice lines until their meaning has been investigated.

## Initial purchase cleansing

From the repository root, run:

```bash
uv run --locked python -m online_retail.data.preprocess
```

This reads **both** workbook sheets and creates `data/interim/purchases.parquet`
plus `data/interim/purchases.report.json` (both ignored by Git). The report records
source SHA-256, input/output sizes, missing values, duplicate counts, and sequential
removal counts. Rules overlap in the source; each discarded row is counted only
under the first matching rule, so counts will differ from standalone EDA counts.

Exclude missing customer IDs, missing essential purchase fields, nonfinite
quantity/price, cancellation-prefixed invoices (case-insensitive), quantities
or prices <= 0, and stock codes TEST001, TEST002, ADJUST2, BANK CHARGES.
These stock-code decisions are explicit assumptions from the initial audit.
Keep missing descriptions and countries including Unspecified.
After filtering, consolidate lines whose fields all match except Quantity by
summing Quantity. Different prices, descriptions, dates, customers, or countries
remain separate. Missing descriptions participate in grouping. This preserves
recorded quantity and sales value; it does not establish the true quantity sold.
The report separates excluded rows from rows consolidated by aggregation.

Output types: Invoice, StockCode, Description, Customer ID, and Country use
pandas nullable strings; Quantity is int64; InvoiceDate is datetime64; Price
is float64. Fractional purchase quantities raise an error instead of truncating.
The workbook column name `Customer ID` is retained for compatibility.
Do not group countries, encode features, or aggregate customer records yet.
Positive-price postage or gift codes are retained unless explicitly excluded;
zero-price gifts are already removed by the price rule. Code length is not a filter.

The loader validates column names and the cleaner checks that retained invoices
map to one customer each. Invalid numeric/date strings raise an error rather than
being silently converted into missing values. Raw cancellations remain available
in the workbook for a later returns/cancellation analysis.

A row is a recorded invoice line, not necessarily a unique product within an
invoice. Customer frequency must eventually count distinct invoices, not rows.
Date quartiles alone do not establish seasonality: compare monthly counts with
coverage and the incomplete first/last months before drawing that conclusion.
