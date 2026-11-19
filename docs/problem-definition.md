# Customer repurchase prediction

Every Monday at midnight, score every identified customer with at least one
qualifying purchase strictly before the cutoff. Include sparse histories and
long-inactive customers. Use all observed past history; do not impose a recent
activity eligibility threshold. The scoring cadence is weekly and the prediction horizon is 50 days.

One example is `(Customer ID, cutoff)`, not one permanent row per customer.
Customer ID and cutoff are keys, never classifier inputs. New customers can be
scored after their first purchase even when absent from training.

## Target: RepurchaseWithin50Days

Measure elapsed time from the scoring cutoff, not the previous purchase:

| Class | Outcome |
|---|---|
| 0 | No qualifying purchase in `[cutoff, cutoff + 50 days)` |
| 1 | At least one qualifying purchase in `[cutoff, cutoff + 50 days)` |

The horizon is exactly 50 days: a purchase at cutoff counts; a purchase exactly
at cutoff + 50 days does not. The positive probability is the chance of at least
one purchase, not purchase timing or incremental response to marketing.

Class 0 includes late returners and customers who never return. It is not permanent
churn. Every labeled snapshot needs 50 complete days of follow-up. Exclude the
source's final calendar day conservatively and omit incomplete-label snapshots.
Use source timestamps without assuming a verified timezone; coverage is assumed
continuous. For production, observation coverage must come from ingestion metadata.

## Chronological evaluation

Dates are in `configs/prediction.toml` and consumed by the notebook.

- History starts in December 2009. Monday snapshots begin March 1, 2010.
  This warm-up does not require individual customers to have three months of history.
- Training: snapshot outcome windows must end by December 1, 2010.
- Validation: Monday cutoffs in `[December 1, 2010, April 1, 2011)`.
- Refit dataset: at June 1, 2011, collect earlier snapshots with fully known labels.
  The final experiment reserves its last six weeks for policy selection; see modeling.md.
- Test: Mondays on/after June 1, 2011, limited by complete source follow-up.

Refit intentionally overlaps training/validation. It is not an independent split.
Intermediate snapshots may be absent from the evaluation splits but available for
refit when mature. Their transactions are never discarded from feature history.
The fixed final model is evaluated across the test period without weekly retraining.

Within training, later tuning must use date-grouped expanding folds with all training
outcomes known before each fold's validation start. Refit learned preprocessing and
RFM segmentation inside each fold. Do not reuse the initial full-training segmenter
for earlier internal folds. Same customers and overlapping historical windows may
appear across chronological partitions. Weekly examples are correlated.

## Cleaning and interpretation

Use `data/interim/purchases.parquet`: identified positive-quantity/positive-price
purchases, excluding cancellation invoices and explicit adjustment/test stock codes.
Repeated lines with matching non-quantity fields have summed quantities. This preserves
recorded sales; it does not prove repetitions were legitimate. Raw workbook is immutable.
Returns are not netted from monetary features and do not erase earlier purchase events.

Segmentation summarizes behavior; prediction estimates individual purchase probabilities.
Neither estimates incremental campaign benefit. Assess unseen and sparse-history
customers separately. Marketing uplift needs treatment/response data and experiments.
The final experiment selects XGBoost by internal temporal CV ROC-AUC and freezes
a policy-holdout threshold of 0.62 before test evaluation. Costs are illustrative,
and production performance acceptance criteria remain undecided. See modeling.md
and results.md for the completed protocol, calibration diagnostics and outcomes.

RFM cluster counts are investigated on training snapshots before dataset generation.
Scores use equal-weight normalized component ranks and fitting-set tertile bands.
See features.md for diagnostics, limitations and the current selection.
