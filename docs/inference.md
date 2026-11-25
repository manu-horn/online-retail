# Local batch inference

This workflow scores customers using the frozen model and threshold **0.62**.
It does not train, tune, build target labels or change the original evaluation.
The model/policy files under `artifacts/final_roc_policy/` remain unchanged.

## Commands

Run from the repository root after installing the locked environment and generating
cleaned purchases and the final model (see [README](../README.md)). Package once:

```bash
uv run python scripts/package_model.py --source artifacts/final_roc_policy --output artifacts/bundles/xgboost-v1 --log-file artifacts/logs/package.jsonl
```

Then score one Monday cutoff:

```bash
uv run python scripts/score.py --bundle artifacts/bundles/xgboost-v1 --cutoff 2011-06-06 --coverage-end 2011-12-09 --output artifacts/scoring/replay-2011-06-06 --log-file artifacts/logs/scoring.jsonl
```

The default purchase source is `data/interim/purchases.parquet`; override with
`--purchases`. Output directories must not already exist. The example replay has
already been run locally; use a new output directory to repeat it. Reuse the bundle
rather than packaging over it. Scoring identical inputs in a new directory produces
the same probabilities/decisions, with a new run ID and runtime metadata.

This is a historical replay using the retail dataset, not a live campaign or feed.

## Input contract

- Supply cleaned purchases with string customer/invoice/product identifiers,
  timezone-naive invoice dates, positive integer quantities and positive prices.
  Missing descriptions are fine. Use the existing cleansing pipeline first.
- Include all available observed history, following the training feature definitions.
  Do not supply only a recent activity extract; completeness cannot be inferred
  from purchase events alone.
- `--cutoff` must be a timezone-naive Monday at midnight. Only strictly earlier
  transactions contribute features or customer eligibility. Future rows in the
  source file do not influence features. Sparse/newly observed customers are included.
- `--coverage-end` is a required **exclusive ingestion coverage boundary**. It is
  an assertion about complete source availability, not the last transaction date.
  Production ingestion must supply trustworthy coverage metadata. This first
  interface checks cutoff <= declared boundary; it cannot prove there are no gaps.
- Cutoffs must follow completed final-fit and policy outcome windows. The packaged
  policy becomes available after its last 50-day follow-up window; the first
  supported weekly replay with the current policy is June 6, 2011.

The existing `customer_features` function constructs the same predictors as
training. Missing purchase gaps stay missing for the fitted imputer. No 50-day
future outcomes are required for inference.

## Bundle and integrity

A new bundle directory contains:

| File | Contents |
|---|---|
| `model.pkl` | Trusted fitted preprocessing/classifier pipeline |
| `frozen_policy.json` | Feature order, threshold and original policy metadata |
| `manifest.json` | Schema, target/horizon, model/policy/feature hashes, availability and packaging environment |

Loading checks model/policy checksums, feature-definition identity, ordered input
schema and binary class order before prediction. Changed feature definitions require
an explicitly rebuilt and validated bundle. The recorded environment is the packaging
environment, not proof of the original training environment. Use the locked project
environment and trusted artifacts. Checksums detect changes; they are not signatures
or an access-control mechanism. This is not a model registry or promotion workflow.

## Scoring outputs

Each new run directory contains `scores.parquet` and `run.json`.

| Score column | Meaning |
|---|---|
| `Customer ID`, `cutoff` | Unique customer/scoring-date key |
| `purchase_probability` | Probability of purchase in the next 50 days |
| `contact` | Probability >= packaged threshold |
| `model_version`, `policy_version` | SHA-256 identities of the frozen artifacts |
| `feature_version` | Hash of shared feature-definition source files |
| `data_version` | SHA-256 identity of the supplied purchase file |
| `run_id` | Scoring execution identifier |

`run.json` records coverage, versions, horizon, threshold, aggregate counts, missing
feature counts, mean probability and runtime. Artifact/score directories are staged
and renamed after writing; failures do not leave partial published output directories.
Existing destinations are rejected rather than overwritten. No raw customer records
are placed in application logs; customer-level scores remain in the output Parquet.

## Structured logging

The new packaging/scoring commands emit JSON events to stderr. `--log-file` also
appends to a JSON-lines file; keep that file outside the new output directory.
Events include UTC timestamp, severity and event name. Scoring events add run ID
and cutoff; success adds versions, counts and runtime. Failure includes a traceback
and exits nonzero.

Events: `packaging_started`, `packaging_succeeded`, `packaging_failed`,
`scoring_started`, `features_built`, `scoring_succeeded`, `scoring_failed`.
This configures the application logger only. Existing training/MLflow scripts have
not been converted to structured logging, and there is no alerting/monitoring service yet.

## Verification and code

The June 6 replay scored **4,951 customers**, with **682 contacts**. Keys,
probabilities and decisions match the saved final test predictions exactly. This
checks inference parity, not a new model-performance estimate.

Tests cover future-row exclusion, sparse histories, boundary decisions, reproducible
scores, integrity/schema checks, invalid cutoffs, coverage, historical policy maturity,
overwrite protection and JSON failure events. Source code is under
`src/online_retail/serving/`; logging is in `src/online_retail/logging.py`.

Fixed-parameter fitting is now verified; see [training reproduction](training-reproduction.md).
Training and inference containers are now verified locally; see [Docker and CI](containers-ci.md).
Training and inference CI checks also pass remotely.
Next work: broader logging coverage and
the [AWS batch workflow](aws-batch-workflow.md).


SageMaker Batch Transform uses the same frozen pipeline through a separate HTTP
adapter receiving prepared snapshot JSON Lines. The training adapter exports a
compatible bundle under `/opt/ml/model`. Both are verified locally; see
[SageMaker contracts](sagemaker-contracts.md). No AWS job has been launched.


For one-command scoring plus aggregate checks, add `--monitoring-reference` to the
existing scoring command. It writes distribution/missingness reports and captures
the scoring-time history group for later outcome evaluation. See [monitoring](monitoring.md).
