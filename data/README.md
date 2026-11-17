# Data and generated datasets

Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
Citation: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5CG6D. License: CC BY 4.0; preserve attribution.

Data files and fitted artifacts are ignored by Git. Keep the raw workbook immutable.

## Files

| Location | Contents |
|---|---|
| `raw/online_retail_II.xlsx` | Original workbook; both year worksheets are read |
| `interim/purchases.parquet` | Cleaned positive purchase lines, with repeated matching lines consolidated |
| `interim/purchases.report.json` | Source checksum, exclusion counts, aggregation counts and quality checks |
| `interim/customers.parquet` | Exploratory full-history customer table, when saved from notebook 02 |
| `processed/train.parquet` | Initial training customer-week snapshots |
| `processed/validation.parquet` | Later snapshots, scored with train-fitted RFM segmentation |
| `processed/refit.parquet` | All mature pre-June snapshots for final model fitting; overlaps train/validation |
| `processed/test.parquet` | Final held-out snapshots, scored with refit-fitted RFM segmentation |
| `processed/train_rfm.pkl` | Initial training-fitted component clusters and score boundaries |
| `processed/refit_rfm.pkl` | Final refit-fitted clusters and score boundaries |
| `processed/rfm_diagnostics.json` | Training-only candidate cluster counts, inertia, silhouette and group sizes |
| `processed/manifest.json` | Source hash, observation boundary, settings, schema roles, sizes and RFM fit details |

Pickle files must only be loaded from trusted sources. The manifest is the authority
for the generated dataset version; counts below describe the current build.

## Cleansing

From the repository root:

```bash
uv run --locked python -m online_retail.data.preprocess
```

Read both workbook sheets; exclude missing customer IDs, missing essential purchase
fields, nonfinite quantities/prices, cancellation-prefixed invoices, quantity or
price <= 0, and stock codes TEST001, TEST002, ADJUST2, BANK CHARGES.
Keep missing descriptions and recorded countries. Stock-code length is not a filter.

Sum quantities when all other line fields match; differing prices or metadata stay
separate. This preserves recorded sales value without resolving whether source
repetitions were legitimate. Exclusion counts are sequential and disjoint;
consolidation counts are recorded separately. Raw returns remain in the workbook.

Text columns use nullable strings; Quantity is int64, Price float64, InvoiceDate
datetime. Invalid numeric/date values and fractional purchase quantities raise errors.
The column name `Customer ID` is retained. Purchase value excludes returns and is
not net retailer revenue. Records without identified customers are outside modeling.

## Processed snapshot contract

Run the RFM diagnostics and Feature Engineering sections of
[`02_stats_and_fe.ipynb`](../notebooks/02_stats_and_fe.ipynb) to regenerate processed
files. Dates come from `configs/prediction.toml`. All non-RFM features are built
point-in-time; RFM cluster-count selection uses training snapshots only.

Each row is one `(Customer ID, cutoff)` pair. Every Monday, include all customers
with any qualifying purchase strictly before the midnight cutoff. Use all observed
history plus 30/90-day activity. Include sparse and long-inactive customers.

`RepurchaseWithin50Days` measures the purchase occurrence relative to cutoff:

| Class | Outcome |
|---|---|
| 0 | No qualifying purchase in `[cutoff, cutoff + 50 days)` |
| 1 | At least one qualifying purchase in `[cutoff, cutoff + 50 days)` |

Only fully observed labels are retained. Observation ends exclusively at
2011-12-09 00:00 in the current build, conservatively excluding the last partial
day. Class 0 is not permanent churn. Source coverage is assumed continuous.

| Dataset | Rows | Distinct customers | Monday cutoffs |
|---|---:|---:|---|
| Train | 92,319 | 3,687 | 2010-03-01–2010-10-11 |
| Validation | 75,608 | 4,693 | 2010-12-06–2011-03-28 |
| Refit | 205,510 | 4,753 | 2010-03-01–2011-04-11 |
| Test | 104,034 | 5,558 | 2011-06-06–2011-10-17 |

Training labels finish by December 1, 2010. Refit labels finish by June 1, 2011.
Refit is not an independent evaluation split. Gaps between cutoff ranges enforce
label availability; their earlier transactions remain usable as feature history.
Customers may occur across dates and splits. Weekly observations are correlated.

## Feature and scoring conventions

The current schema has 27 predictors. Use `manifest["feature_columns"]` to select
them; never include Customer ID, cutoff or RepurchaseWithin50Days in classifier inputs.
Groups include historical RFM, recent activity, basket metrics, purchase-day gaps,
history indicators, annual calendar features and relative RFM scores.
Missing gaps represent insufficient history and remain missing; do not replace them
with zero. First observed purchase does not establish true customer tenure.

RFM diagnostics compare k=1–6; the current elbow heuristic selects R=2, F=2, M=2.
Ordered component ranks are normalized to 0–100 and equally averaged. Score tertiles
learned from the fitting partition define low/medium/high relative segments. Ties
stay together, so groups need not be equally sized. Validation uses the train fit;
test uses the refit fit, with the selected cluster counts unchanged.

See [feature definitions](../docs/features.md) and
[prediction contract](../docs/problem-definition.md) for the full decisions.

## Analysis before modeling

[`03_eda_processed.ipynb`](../notebooks/03_eda_processed.ipynb) explores training
features and outcomes. It does not load validation, refit or test tables. Pooled
plots describe customer-week instances; single-cutoff plots describe distinct
customers. Associations are descriptive, not causal or validated model performance.
Keep the final test untouched. Learn imputation, encoding and any feature selection
inside chronological training folds when model building begins.
