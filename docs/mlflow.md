# MLflow integration

This branch adds local experiment tracking and executable training/evaluation
scripts. Temporal folds, model selection and the frozen final policy are unchanged.
The reference repository's logging pattern is adapted to our existing pipelines;
its random splits are not copied.

## Structure

| Location | Responsibility |
|---|---|
| `src/online_retail/modeling/tracking.py` | MLflow configuration, provenance, metrics, models and artifact imports |
| `src/online_retail/modeling/workflows.py` | Training/evaluation orchestration shared by scripts and notebooks |
| Existing model modules | Estimators, preprocessing, temporal folds and metrics |
| `scripts/train.py` | Fresh tracked development training |
| `scripts/track_results.py` | Import saved experiment outputs without training |
| `scripts/evaluate.py` | Frozen-model diagnostics, tables and PNG plots |
| `configs/tracking.toml` | Experiment name; storage defaults are repository-local |

Scripts are thin entrypoints; reusable logic remains in `src/`. Existing functions
can still be called without tracking. Notebook 04's training calls now use the
tracked workflow. Notebooks 04/05 include optional saved-report import cells;
those switches default to false. Existing notebook outputs predate this integration
and are not evidence that a tracked training run has been repeated.

## Local setup and commands

From the repository root:

```bash
uv sync --locked --group notebooks
uv run python scripts/train.py baseline --output artifacts/runs/baseline-001
uv run python scripts/train.py elastic_net --output artifacts/runs/elastic-001
uv run python scripts/train.py trees --trials 15 --objective weekly_roc_auc --output artifacts/runs/trees-001
```

Training expects the processed datasets from notebook 02. Scripts require a new
output directory to protect earlier artifacts. Training does not invoke `run_final`
or select anything on test. MLflow logging completes after model/report generation;
a logging failure marks the workflow FAILED but does not roll back generated files.

To explore existing results without repeating training or final evaluation:

```bash
uv run python scripts/track_results.py artifacts/final_roc_policy --name frozen-xgboost-import
uv run python scripts/track_results.py artifacts/roc_auc_models --name roc-auc-study-import
uv run python scripts/evaluate.py --output artifacts/runs/evaluation-001
```

Run the import commands as separate shell commands. Do not put a trailing `\`
after the first command: it would join the second command's arguments to it.
To chain them intentionally, use `&&`:

```bash
uv run python scripts/track_results.py artifacts/logistic_baseline --name baseline-import &&
uv run python scripts/track_results.py artifacts/logistic_elastic_net --name elastic-net-import
```

An import that ends with a run ID and exit status 0 succeeded, even if MLflow
printed schema, serialization or environment warnings. The UI shows its status
as FINISHED. Repeating an import creates another tracking run; it does not retrain.

The evaluation script reproduces frozen probabilities and descriptive test slices,
exports figures and reports, and records the source model hash and policy. It does
not retrain, recalibrate or change thresholds. Test has already been inspected.
SHAP remains in notebook 05; the evaluation script covers the core metric/group plots.

Open the UI using the same backend, from the repository root:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Visit http://127.0.0.1:5000. Local defaults are an absolute SQLite URI for
`mlflow.db`, file artifacts in `mlartifacts/`, and experiment
`online-retail-repurchase`. `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` override
configuration. Existing experiment artifact locations are retained. HTTP tracking
servers use their own default artifact location when creating an experiment.
Database/artifact files are Git-ignored. No persistent UI or cloud server is
started automatically.

## What gets logged

- Git commit, branch and dirty status; a source archive captures current package
  and script files, including uncommitted changes.
- Dataset checksums, the processed manifest, split/tracking configuration, package
  versions, pyproject and uv lockfile.
- Selected parameters, validation metrics, fold metrics and completed trial objectives.
- CSV/JSON/PNG reports, fitted preprocessing plus estimator, input/output signature,
  feature order and a five-row engineered-feature input example.
- Final policy metadata and existing reported test results when explicitly importing
  the final artifact directory. The threshold is not part of model prediction.

Raw datasets and prediction Parquets are not uploaded. Artifact imports are tagged
`historical_artifact_import`: code/environment/checksums describe logging time,
not proof of the original training revision. Training runs wrap actual execution;
failures become FAILED runs. Imports never modify source model files or rerun test.

A tree workflow has model-summary child runs and completed-trial grandchildren.
Each trial summary records its parameters, objective and fold metrics at steps
1–3. Their durations are logging durations, not original trial training times.
Detailed trial/fold tables remain attached as CSVs. No autologging is used, avoiding
ambiguous automatic metrics across many CV fits. No registry promotion is performed.

## Loading a logged model

The run's `model.uri` tag identifies its logged model. The MLflow pyfunc flavor
returns **two probability columns**, ordered no-purchase then purchase. Native
scikit-learn loading retains the pipeline's usual `predict`/`predict_proba` methods.

```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
run = MlflowClient().get_run("<run-id-containing-a-model>")
model = mlflow.pyfunc.load_model(run.data.tags["model.uri"])
probabilities = model.predict(customer_features_in_saved_column_order)
purchase_probability = probabilities[:, 1]
```

The model includes fitted preprocessing but not transaction-to-snapshot feature
construction or campaign decisions. Use the corresponding frozen policy artifact
for threshold/cost assumptions. Load trusted artifacts using the project environment.
This integration provides tracking, not a production inference service.

## Verification

Integration tests train a small chronological fixture, verify metrics/provenance,
reload a logged model and compare probabilities, import artifacts without mutation,
and confirm failed runs close correctly. The original evaluated model remains
unchanged. AWS jobs, registry approvals and production scheduling are later roadmap
milestones; see [AWS workflow](aws-batch-workflow.md).

## Fixed-parameter reproduction

`scripts/reproduce_fit.py` creates an `xgboost-fixed-fit-reproduction` run with
workflow mode `fixed_parameter_reproduction`. It trains anew from the reference
pipeline settings, retains the frozen threshold, and compares probabilities on
the policy holdout. It does not run Optuna or evaluate test rows. The new model
and parity/provenance report are logged separately from the evaluated reference.
See [commands and evidence](training-reproduction.md).
