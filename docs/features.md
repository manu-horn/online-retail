# Feature engineering

Run the Feature Engineering section of `notebooks/02_stats_and_fe.ipynb` after
loading purchases. Helpers live in `src/online_retail/features/`. Training-only RFM diagnostics precede dataset building. Processed-feature analysis lives in notebook 03 and the binary logistic baseline
in notebook 04.

Every feature uses transactions strictly before its Monday cutoff. No customer
is dropped for insufficient purchase-gap history. The source starts in December
2009: observed-history length is not true tenure, and totals are not lifetime totals.

| Feature | Definition |
|---|---|
| recency_days | Fractional elapsed days since latest historical invoice |
| invoice_count | Distinct invoices over all available history |
| monetary_total | Sum of Quantity × Price, excluding returns |
| observed_history_days | Fractional days since first observed purchase |
| purchase_day_count | Number of distinct historical calendar purchase days |
| invoice_count_30d / 90d | Distinct invoices in `[cutoff − window, cutoff)` |
| monetary_30d / 90d | Purchase value over the corresponding recent window |
| average_invoice_value | Historical value / invoice count |
| average_units_per_invoice | Historical quantity / invoice count |
| distinct_product_count | Distinct stock codes in historical purchases |
| last_gap_days | Latest minus previous distinct purchase day |
| gap_2_span_days / gap_3_span_days | Latest minus purchase day two/three events earlier |
| mean_gap_days / std_gap_days | Mean and sample std of consecutive purchase-day gaps |
| has_previous_purchase_day | At least two historical purchase days |
| has_four_purchase_days | At least four historical purchase days |
| month_sin / month_cos | Cyclic month: angle `2π(month−1)/12` |
| days_until_christmas | Calendar days to next December 25, including today |

Gap values remain missing if unavailable. Std needs two completed gaps. Multiple
same-day invoices count separately for frequency but form one purchase day for gaps.
We do not save historical timestamp auxiliaries. Raw timestamps, IDs, cutoff and
target are excluded from the saved classifier-column list. Encoding/imputation for
classifier fitting will be handled later inside its training pipeline.

## RFM clustering diagnostics and scoring

The notebook's RFM analysis runs before feature engineering, using only training
snapshots. Compare k=1–6 separately for recency, log1p(invoice count), and
log1p(monetary total). Inertia uses all training rows; silhouette uses a fixed
sample of at most 1,500 rows. Show smallest-cluster shares alongside both curves.

Select an initial k using maximum distance below the normalized inertia endpoint
chord, excluding endpoint k values and candidates with any cluster below 1% of
training rows. This reproducible elbow heuristic is not an optimality claim;
inspect profiles, centers and silhouette. The current source recommends R=2,
F=2, M=2. Frequency silhouette keeps increasing as discrete counts split more
finely, so maximizing silhouette alone would choose a different granularity.
The candidate range and population affect the recommendation.

Every training customer-week has equal weight. Repeated customers contribute
multiple observations. To visualize the customer base without counting a person
multiple times, profile the last training cutoff, one row per customer. Show
recency/frequency and frequency/monetary scatter plots, segment counts and medians,
and component centers in original units. This does not use prediction labels.

Fit independent one-dimensional KMeans models with selected k (seed 42,
10 initializations). Order centers so higher scores mean lower recency or higher
frequency/spend. Normalize each component rank to 0–100 using rank/(k−1), then
average the three scores equally. This permits different k without giving extra
weight to components with more clusters. The score remains ordinal, not a
calibrated probability or evidence of natural customer types.

Learn combined-score tertiles on the fitting population: score <= first tertile
is low, score > second tertile is high, otherwise medium. Ties stay together;
segment sizes can be unequal or a band empty. If both tertile boundaries coincide,
raise an error and review segmentation rather than arbitrarily splitting ties.
These are relative engagement/value descriptions, not pure activity or targets.

Fit initial centers and score boundaries on train only; transform validation
unchanged. For final refit, keep selected k fixed but relearn centers and score
boundaries on all mature pre-test snapshots. Transform test unchanged. Fold-based
model tuning must repeat cluster selection/fitting within each training fold if
selection is part of the tuned pipeline; never use full-training diagnostics to
claim independent performance on earlier internal folds.

## Outputs

`data/processed/` contains train, validation, refit and test Parquet tables, two
trusted-local pickle segmenters, and a JSON manifest with coverage, source hash,
settings, feature columns, dataset sizes, selected cluster counts, fitted centers
and score boundaries. `rfm_diagnostics.json` records the selection evidence. These are Git-ignored.
Pickles should only be loaded from trusted sources. Target and keys are retained
in tables for supervised training and joins, but are not predictors.

Product categories, country, cancellation features, customer-specific annual
seasonality, remain deferred. The full-history
exploratory customer table is not used to build historical snapshots.
