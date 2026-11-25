# Fixed-parameter training reproduction

This command repeats the selected XGBoost final fit without Optuna, model selection,
threshold tuning or test evaluation:

```bash
uv run --locked python scripts/reproduce_fit.py --output artifacts/reproductions/xgboost-001
```

It requires the generated processed datasets and trusted local artifacts under
`artifacts/final_roc_policy/`. Use a new output directory on each run. Development
installs include the training dependencies; an inference-only install does not.

## What is fitted

The command clones the reference pipeline's constructor settings, discarding its
learned trees and imputation statistics. This preserves the full XGBoost settings
(including seed and thread count) and median imputation configuration, rather than
relying on the subset of tuned parameters alone. The tuned parameters are checked
against the frozen policy before fitting.

Rows come from `refit.parquet`, in their existing order, restricted to cutoffs up
to **January 10, 2011**: **146,415 customer snapshots**. Outcomes mature before the
policy holdout starts on March 7. The policy holdout and the final test are excluded
from fitting. The frozen threshold remains **0.62**.

## Verification and saved files

Prediction parity is checked on the **28,001 policy-holdout snapshots**, March 7–
April 11, against the saved `policy_predictions.parquet`. Customer/cutoff keys must
match. The reference model must also reproduce those saved predictions. The
absolute probability tolerance is fixed at `1e-7`, with no relative tolerance;
contact decisions must match exactly. Test rows are never loaded. The tracking
wrapper hashes processed files for provenance, including the test file; hashing
is not test evaluation.

A successful run writes a new directory containing:

| File | Purpose |
|---|---|
| `model.pkl` | Newly fitted preprocessing/classifier pipeline |
| `frozen_policy.json` | Byte-for-byte copy of the original policy |
| `policy_predictions.parquet` | Reproduced holdout probabilities and contacts |
| `report.json` | Population, parameters, checksums, input fingerprint, versions and parity |
| `mlflow_run.json` | Tracking run ID |

Model outputs are staged and published only after parity succeeds. Invalid
populations, immature labels, changed settings or failed parity stop the run.
Existing destinations and reference artifacts are never overwritten. The new
pipeline is a reproduction artifact, not an automatic replacement of the frozen
production candidate.

MLflow records this as `fixed_parameter_reproduction`, distinct from a historical
import. It logs the new pipeline, reports, source snapshot, Git revision, dirty-tree
status, lockfile and data checksums. Customer-level prediction files stay local.

## Current evidence and limits

The local reproduction matched all **28,001** saved policy predictions with
**zero probability difference** and identical decisions. The final verified run is
saved at `artifacts/reproductions/xgboost-002/`, with MLflow run ID
`7f707c1b50344ca5a07e251c0a4763d1`. These artifacts are local and Git-ignored.
The reference model and policy remain unchanged. This checks repeatability with the current local inputs
and environment; it is not a new performance estimate or a guarantee of identical
results on every platform. The original training run did not save a fitting-input
fingerprint, so the new checksum cannot retroactively prove original data identity.

Tests exercise parity, no test-data reads, no overwrites, reference preservation,
population changes, label maturity, failed-parity publication and MLflow lineage.
The same command is available in a separate [training container](containers-ci.md#build-and-reproduce-training).
SageMaker execution remains a separate next step.


The training container repeated the full fit with the same zero-difference result.
Its output is `artifacts/container-training/xgboost-001/`, in a separate MLflow store.
See [container commands and verification](containers-ci.md).
