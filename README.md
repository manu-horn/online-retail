# Online Retail: customer repurchase prediction

Predict whether an identified customer will purchase in the **next 50 days**, using
[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
Every Monday, build a customer snapshot from earlier purchases. The binary target
is 1 for a purchase in `[cutoff, cutoff + 50 days)` and 0 otherwise. Include sparse
and long-inactive customers; no purchase within the horizon is not permanent churn.

The project currently covers cleansing, customer analysis, feature engineering,
chronological model tuning and a frozen-policy test evaluation. Deployment and
operational MLOps remain future work.

## Current result

**XGBoost** was selected using temporal CV ROC-AUC. After refitting, a separate
policy holdout selected threshold **0.62**, then the frozen model and threshold
were evaluated on test without further selection.

| Test metric | Result |
|---|---:|
| ROC-AUC | 0.7948 |
| Average precision | 0.6504 |
| Log loss | 0.5098 |
| Buyer precision / recall at 0.62 | 0.7077 / 0.4224 |

Under illustrative TN/FP/FN/TP costs of £0/£15/£25/£10, mean test decision cost is
£5.77 per customer-week versus £6.77 for contacting nobody. These are hypothetical
costs, not measured campaign ROI. The final fit reserves policy data and a
label-maturity gap; it does not fit every row in the generated refit table.

The saved model is `artifacts/final_roc_policy/model.pkl`; threshold and feature
order are in `frozen_policy.json` in the same directory. These files are local and
Git-ignored. See the [results and artifact guide](docs/results.md).

## Environment and checks

Use Python 3.12 and uv. From the repository root:

```bash
uv sync --locked --group notebooks
uv run --locked pytest -q
uv run --locked ruff check src scripts tests
uv run --locked ruff format --check src scripts tests
uv run --locked jupyter lab
```

Choose the project's `.venv` kernel. Runtime dependencies include pandas,
scikit-learn, Optuna, LightGBM and XGBoost; `uv.lock` fixes resolved versions.
Development tools install by default; notebook tools use the optional group.
The lint commands above cover package code and tests; exploratory notebook style
is not included in those checks.

## Run the workflow

1. Place the original workbook at `data/raw/online_retail_II.xlsx`.
2. Run `uv run --locked python -m online_retail.data.preprocess` to read both sheets,
   clean purchase lines and save the source checksum and cleansing report.
3. Use the notebooks in order:

| Notebook | Purpose |
|---|---|
| [01_eda](notebooks/01_eda.ipynb) | Raw-data inspection |
| [02_stats_and_fe](notebooks/02_stats_and_fe.ipynb) | Sales/customer analysis, RFM diagnostics and processed snapshot generation |
| [03_eda_processed](notebooks/03_eda_processed.ipynb) | Training-feature distributions, target relationships and multicollinearity |
| [04_modeling](notebooks/04_modeling.ipynb) | Baselines, tuning, calibration diagnostics, policy selection and test evaluation |

Every feature uses past transactions; every label needs 50 complete future days.
Expanding temporal folds preserve scoring dates and label maturity. Logistic
models use fold-fitted transforms/scaling; trees preserve original feature scales.
The final search maximizes mean weekly ROC-AUC with 15 Optuna trials per tree
family. Calibration plots are diagnostics; no probability recalibration was fitted.

Notebook 04 loads an existing final report on rerun. Its ROC-AUC search has a
separate rerun switch; earlier experiment cells can still retrain. Test has already
been evaluated: do not retune using its outcomes and reuse it as an untouched final
estimate. Keep model/threshold selection on development data.

## Project map

| Location | Contents |
|---|---|
| `configs/prediction.toml` | Snapshot and chronological partition dates |
| `src/online_retail/data/` | Loading and cleansing |
| `src/online_retail/analysis/` | Reusable exploratory plots and diagnostics |
| `src/online_retail/features/` | Past-only snapshots, labels, RFM and dataset building |
| `src/online_retail/modeling/` | Preprocessing, models, temporal folds, metrics and final policy |
| `scripts/` | Thin entrypoints for tracked training, saved-result imports and evaluation |
| `configs/tracking.toml` | MLflow experiment configuration |
| `tests/` | Cleansing, feature-boundary, temporal, metric, policy and tracking checks |
| `data/` | Raw/interim/processed data, Git-ignored except its README |
| `artifacts/` | Fitted models, search reports, predictions and frozen policy; Git-ignored |

## Documentation

- [Prediction contract](docs/problem-definition.md): population, target and time boundaries.
- [Data guide](data/README.md): provenance, cleansing and dataset schema.
- [Feature definitions](docs/features.md): behavioral features and RFM segmentation.
- [Modeling workflow](docs/modeling.md): trials, CV, calibration, refitting and threshold selection.
- [Results and saved artifacts](docs/results.md): final metrics, file meanings and loading example.
- [Evaluation notebook guide](docs/evaluation.md): weekly/group metrics, SHAP and interpretation.
- [MLflow guide](docs/mlflow.md): tracked scripts, local UI and saved-model loading.
- [MLOps roadmap](docs/roadmap.md): completed stages and next work.
- [AWS batch workflow](docs/aws-batch-workflow.md): phased local-to-cloud implementation plan.
[Notebook 05](notebooks/05_evaluation.ipynb) evaluates the frozen model and threshold:
overall curves/calibration diagnostics, weekly performance, past-invoice groups and
customers seen/unseen in the actual final fit. It verifies saved predictions and
exports tables to `artifacts/evaluation/`, without training or threshold selection.
See [diagnostic findings](docs/results.md#frozen-model-diagnostic-evaluation).

Notebook 05 uses the standard SHAP library for group beeswarms and individual
waterfall examples on a reproducible test-snapshot sample. Contributions are
checked against frozen predictions; they explain model behavior, not causal
effects. Tables are saved in `artifacts/evaluation/shap/`.

## Next implementation milestone

Follow the [AWS batch workflow roadmap](docs/aws-batch-workflow.md). Start with local
training/scoring commands, then versioning/MLflow, containers/CI and a small
S3 → SageMaker Training → Batch Transform demonstration. Weekly inference is
separate from retraining. AWS resources and orchestration are planned, not yet
implemented; interactive services are deferred.

## MLflow and executable workflows

Reusable orchestration and logging live in `src/online_retail/modeling/`; `scripts/`
provides thin entrypoints. From the repository root:

```bash
uv run python scripts/track_results.py artifacts/final_roc_policy --name frozen-xgboost-import
uv run python scripts/train.py baseline --output artifacts/runs/baseline-001
uv run python scripts/evaluate.py --output artifacts/runs/evaluation-001
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Run each shell command separately. To chain commands, use `&&`; a trailing `\`
alone continues the same command and can cause an argument-parsing error.

Visit http://127.0.0.1:5000. Runs record parameters, metrics, model/schema, reports,
code/environment and data checksums. Tree searches include model and completed-trial
summary runs. Imports log existing results without retraining or reselecting the
policy. Models expose probabilities; thresholds remain separate artifacts.
SQLite metadata (`mlflow.db`) and `mlartifacts/` are local and ignored by Git.
See the [MLflow guide](docs/mlflow.md) for configuration, provenance limits and loading.
