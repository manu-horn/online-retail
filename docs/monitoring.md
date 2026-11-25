# Local batch monitoring

This workflow runs feature preparation, frozen-model scoring and aggregate monitoring
without AWS or new dependencies. It preserves the model and threshold **0.62**.
Drift reports need no future outcomes; performance evaluation is a separate command
that requires complete 50-day follow-up. The implementation is in
`src/online_retail/monitoring/`.

## 1. Freeze the development reference once

```bash
uv run --locked python scripts/create_monitoring_reference.py --bundle artifacts/bundles/xgboost-v1 --output artifacts/monitoring/reference-policy-apr11
```

The reference is **4,753 customers at April 11, 2011**, the last policy-holdout
cutoff recorded by this bundle. Each customer appears once. The source is
`data/processed/refit.parquet`, read with only keys/features, not target columns.
This is a development distribution reference, not a model-performance benchmark.
It stores aggregate summaries/histograms and no customer-level records.

The reference records its source checksum, model/policy/feature identities,
feature order and limits from `configs/monitoring.toml`. Keep it fixed for comparable
reports. A different model/policy/feature version needs a new reference. Updating
reference populations or limits is an explicit versioned decision, not an automatic
response to warnings. Existing destinations are rejected.

## 2. Run a monitored scoring batch

```bash
uv run --locked python scripts/score.py --bundle artifacts/bundles/xgboost-v1 --cutoff 2011-06-06 --coverage-end 2011-12-09 --monitoring-reference artifacts/monitoring/reference-policy-apr11/reference.json --output artifacts/scoring/monitored-2011-06-06 --log-file artifacts/logs/monitoring.jsonl
```

The default source is cleaned `data/interim/purchases.parquet`. Cutoff and coverage
rules from [local inference](inference.md) still apply. Use a new output directory
to repeat these examples; the paths shown have already been generated locally.

A successful run publishes:

| File | Contents |
|---|---|
| `scores.parquet` | Existing scores/versions plus history group at the scoring cutoff |
| `run.json` | Counts, contact rate, mean score, missingness, runtime, versions and history fingerprint |
| `monitoring.json` | Aggregate comparisons, reference identity, settings and warning codes |
| `monitoring.md` | Small human-readable table and interpretation |
| `monitoring.png` | Probability distribution and customer-history mix |

The default command without `--monitoring-reference` keeps its existing behavior.
The Docker inference image accepts the same optional flag when the reference file
is mounted read-only. Monitoring uses the already constructed features; it does
not calculate a second feature table or refit anything.

Missingness is recorded for every predictor. Histogram drift checks focus on
recency, invoice count, spending, recent invoice counts, average invoice value,
last purchase gap and observed history length. Calendar-only predictors are not
used for drift warnings. Probability and history-group distributions are also
compared. Sparse-history customers stay included; missing gaps remain missing
for the saved imputer.

### How warnings work

Histogram **total variation distance** is half the sum of absolute bin-share
differences: 0 means identical histogram shares; 1 means disjoint occupied bins.
Feature bin boundaries are frozen reference quantiles, with separate below-range,
above-range and missing bins. Constant/all-missing reference features are supported.
Probability bins are fixed at 0.1 increments. No significance tests or independent-
sample assumptions are used for these repeated customer populations.

Initial limits are review heuristics, not validated operational alarms:

| Check | Warning rule |
|---|---|
| Feature/probability/history-group distribution | TV distance > 0.20 |
| Predictor missingness | Increase > 5 percentage points |
| Contact rate | Absolute change > 5 percentage points |
| Population | < 100 rows or < half the reference population |

Warnings do not block otherwise valid scores or trigger retraining. Invalid data,
bundle/reference mismatches or report failures produce a nonzero exit and no new
published output directory. JSON logs retain `scoring_failed` with run ID/cutoff.
Successful comparisons emit `monitoring_completed` with status and warning codes.
`--log-file` preserves the events locally; stderr is also available to a scheduler.
Logs contain aggregate context rather than customer records.

The June 6 replay scored **4,951 customers**, with **682 contacts**, matching the
existing predictions. It warned on `observed_history_days` (TV about 0.325): time
and customer composition have changed since April. Review that context before
interpreting the warning; drift alone does not establish reduced model quality.
The limits were not adjusted to make this historical replay pass without warnings.

## 3. Evaluate only when outcomes mature

```bash
uv run --locked python scripts/evaluate_batch_outcomes.py --run artifacts/scoring/monitored-2011-06-06 --coverage-end 2011-12-09 --output artifacts/monitoring/outcomes-2011-06-06 --log-file artifacts/logs/monitoring.jsonl
```

This uses the saved scored customer population, not every customer in the follow-up
file. A buyer has a qualifying purchase in `[cutoff, cutoff + 50 days)`. Coverage
must reach the exclusive end of that window. A customer with no observed purchase
is not labeled a negative before follow-up is complete. As with scoring, declared
coverage is trusted ingestion metadata; transaction timestamps alone cannot prove it.

`report.json` and `report.md` contain ROC-AUC, average precision, log loss, Brier
score, accuracy, buyer precision/recall/F1 and confusion matrices, overall and for
1, 2–3 and 4+ observed invoices. The history group is saved at prediction time.
Undefined AUC/AP are null, not fabricated zeros. Precision/recall/F1 use zero when
their denominator is zero; consult group sizes/buyer counts before interpreting them.
Each report is one cutoff/week. Weekly populations overlap and are not independent.

Follow-up must be cleaned purchases. A normalized, order-independent fingerprint
checks that pre-cutoff purchase history has not changed. Adding future observations
or reordering rows is allowed; historical backfills require explicit reconciliation.
Saved score counts, identities, probabilities and decisions are checked too.
The fingerprint detects ordinary input changes; it is not a digital signature.

This is descriptive monitoring of a historical replay, not new model selection,
threshold tuning or proof that a campaign causes purchases. The historical test
period has already been inspected; future model changes need a new evaluation plan.

## Optional MLflow import

Keep inference free of the MLflow runtime dependency. On the host, existing tracking
commands can import the completed aggregates without training or uploading customer
predictions:

```bash
uv run --locked python scripts/track_results.py artifacts/scoring/monitored-2011-06-06 --name monitoring-june-06
uv run --locked python scripts/track_results.py artifacts/monitoring/outcomes-2011-06-06 --name outcomes-june-06
```

The example imports have already been run locally as `monitoring-june-06`
(run `01d69ce1aa14467d9b9748056d061b24`) and `outcomes-june-06`
(run `b27825c77be74cf4a7e3b56db206c85c`). Repeating an import creates another run.
Monitoring metrics use the `monitoring.` prefix; lineage
tags link the scoring run, reference and source model. Outcome metrics use the
existing `reported.` prefix. No model is copied or promoted by these imports.

## Scope

The existing reports, logs and mature-outcome checks are sufficient for the initial
batch demonstration. Warning limits remain documented review heuristics; they
are not validated production alarms. Run the commands manually for chosen replay
cutoffs. No unattended alerting service is claimed.

Weekly trend indexes, scheduling, notifications and registry/promotion/rollback
automation are optional later extensions, not outstanding requirements for this
milestone. See the [simple roadmap](roadmap.md).


## Verification

All 61 tests pass locally. The inference image also passes its extended smoke
check with monitored scoring enabled (`artifacts/ci/monitoring-container.json`).
Scoring probabilities and decisions remain unchanged; warning reports require
no labels. Tests cover fixed/constant reference bins, missingness, stale references,
failure logging without published output, mature coverage boundaries, cohort
identity, historical revisions and MLflow aggregate imports. The extended container check is included in CI; consult the branch
Actions run for its current remote status.
