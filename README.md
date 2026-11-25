# Online Retail: customer repurchase prediction

**Predict whether a customer will purchase again within the next 50 days**, using
historical transactions from [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
The goal is to support customer targeting while building a reproducible,
batch-oriented MLOps workflow.

Every Monday, each identified customer with a previous qualifying purchase gets
one **snapshot**: features summarizing their history at that cutoff. The target is
**1** if they purchase in `[cutoff, cutoff + 50 days)`, otherwise **0**. Customers
with little history or long inactivity remain eligible. No purchase within this
window does not mean permanent churn.

## Main results

**XGBoost** was selected by mean weekly ROC-AUC across expanding temporal
cross-validation folds. After refitting, a later development policy holdout selected
threshold **0.62**; the model and threshold were frozen before test evaluation.

Results below cover **104,034 test snapshots from 5,558 distinct customers**, with
weekly cutoffs **June 6–October 17, 2011**. Customers can appear in multiple weeks.

| Test metric | Result |
|---|---:|
| ROC-AUC | 0.7948 |
| Average precision | 0.6504 |
| Log loss | 0.5098 |
| Brier score | 0.1673 |
| Buyer precision at 0.62 | 70.77% |
| Buyer recall at 0.62 | 42.24% |
| Accuracy at 0.62 | 79.62% |

This is a selective policy: it targets about 16.2% of snapshots and reaches about
42% of eventual buyers. Performance differs substantially with purchase history:

| Invoices observed before cutoff | ROC-AUC | Buyer recall at 0.62 |
|---|---:|---:|
| 1 | 0.711 | 0% |
| 2–3 | 0.651 | 1.8% |
| 4+ | 0.767 | 60.2% |

Under illustrative TN/FP/FN/TP costs of £0/£15/£25/£10, mean test decision loss is
**£5.77 per customer-week**, versus **£6.77** for contacting nobody. These are
assumed penalties, not measured campaign profit. Purchase propensity does not
establish that advertising causes additional purchases.

See [full results](docs/results.md), [evaluation](docs/evaluation.md) and the
[worked cost example](docs/modeling.md#worked-cost-example-threshold-062).

## Current project status

The intended scope is **data preparation → training → MLflow tracking → saved
model → batch predictions → basic monitoring**.

| Part | Status |
|---|---|
| Data analysis, past-only features and temporal model evaluation | Implemented |
| MLflow tracking and fixed-parameter training reproduction | Implemented locally |
| Versioned bundles, batch scoring and JSON logs | Implemented locally; predictions match the saved replay |
| Docker images and GitHub Actions code/container checks | Implemented |
| Basic drift reports and mature-outcome monitoring | Implemented locally |
| S3 → SageMaker Training → Batch Transform replay | Adapters verified locally; AWS access/cost checks unresolved |

The preferred cloud budget is **$0** and the configured AWS account is currently
closed. No cloud resources have been created by this workflow. The main remaining
milestone is one small AWS replay if access and costs allow it.

Scheduling, dashboards, notifications, registry promotion, automated rollback and
automatic retraining are optional extensions, not requirements for this project.
See the [simple roadmap](docs/roadmap.md), [monitoring guide](docs/monitoring.md)
and [AWS replay plan](docs/aws-batch-workflow.md).

## Set up the environment

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --locked --group notebooks
uv run --locked pytest -q
uv run --locked ruff check src scripts tests
uv run --locked ruff format --check src scripts tests
uv run --locked jupyter lab
```

The default development group includes training tools. `uv sync --locked --no-dev`
installs inference dependencies only; add `--group training` for training without
development tools.

Select the project's `.venv` notebook kernel. `uv.lock` fixes dependency versions;
tools include scikit-learn, Optuna, XGBoost, LightGBM, SHAP and MLflow. Lint and
format checks above cover package code, scripts and tests, not exploratory notebooks.

## Reproduce from a fresh clone

Raw data, generated datasets, model artifacts and the MLflow database are **not
included**. To reproduce the workflow:

1. Download the dataset from UCI and place the extracted workbook at
   `data/raw/online_retail_II.xlsx`. Preserve the original file.
2. Run `uv run --locked python -m online_retail.data.preprocess` to read both
   worksheets and generate cleaned purchases and a provenance/cleansing report.
3. Follow the notebooks in order using the project kernel:

| Notebook | Purpose |
|---|---|
| [01_eda](notebooks/01_eda.ipynb) | Raw-data inspection |
| [02_stats_and_fe](notebooks/02_stats_and_fe.ipynb) | Sales/customer analysis, RFM diagnostics and snapshot datasets |
| [03_eda_processed](notebooks/03_eda_processed.ipynb) | Training-feature relationships and multicollinearity |
| [04_modeling](notebooks/04_modeling.ipynb) | Baselines, Optuna searches, policy selection and frozen test evaluation |
| [05_evaluation](notebooks/05_evaluation.ipynb) | Overall/weekly/group evaluation and SHAP beeswarms/waterfalls |

Features use only transactions before each cutoff. Training outcomes must mature
before the next validation period; random row splits are not used. Final fitting
reserves policy-holdout weeks and a 50-day maturity gap, so it does not fit every
row in the generated refit pool. See the [modeling protocol](docs/modeling.md).

Notebook 04 loads an existing final report on rerun. Its earlier cells can retrain;
the ROC-AUC search has a separate rerun switch. Existing notebook outputs document
the reported experiment. We have already inspected test results, including further
diagnostics: future model/policy changes require a new evaluation plan.

## Use existing local artifacts and MLflow

The following commands require the generated files above. They are **not**
fresh-clone setup commands. Importing saved results does not retrain:

```bash
uv run python scripts/track_results.py artifacts/final_roc_policy --name frozen-xgboost-import
uv run python scripts/evaluate.py --output artifacts/runs/evaluation-001
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Run each command separately. Open http://127.0.0.1:5000 for the tracking UI.
Imports create new runs; do not repeat them unless another import is intended.
Training is available through `scripts/train.py`, using processed datasets and a
new output directory. See the [MLflow guide](docs/mlflow.md) for commands and lineage.

The final fitted pipeline is saved at `artifacts/final_roc_policy/model.pkl`, with
its feature order and threshold in `frozen_policy.json`. These files are generated
locally, not downloadable from this repository. MLflow stores probabilities and
models separately from the contact policy. Its local database and artifacts remain
Git-ignored.

## Project map and documentation

| Location | Contents |
|---|---|
| `src/online_retail/data/` | Loading and cleansing |
| `src/online_retail/features/` | Snapshot features, labels, RFM and dataset generation |
| `src/online_retail/analysis/` | Exploratory plots and diagnostics |
| `src/online_retail/modeling/` | Models, temporal evaluation, explanations, workflows and tracking |
| `src/online_retail/serving/` | Frozen bundles and batch inference |
| `src/online_retail/monitoring/` | Development reference, drift reports and mature outcomes |
| `src/online_retail/sagemaker/` | Training paths, prepared JSON Lines and Batch Transform HTTP adapter |
| `scripts/` | Training, evaluation, tracking, packaging and scoring entrypoints |
| `configs/` | Prediction dates and tracking settings |
| `tests/` | Data, temporal, metric, policy, explanation and tracking checks |
| `docs/` | Version-controlled project guides |
| `data/`, `artifacts/` | Local generated files; ignored except `data/README.md` |

- [Prediction contract](docs/problem-definition.md)
- [Data provenance and cleansing](data/README.md)
- [Feature definitions and RFM](docs/features.md)
- [Modeling workflow](docs/modeling.md)
- [Results and saved artifacts](docs/results.md)
- [Evaluation notebook guide](docs/evaluation.md)
- [Local inference guide](docs/inference.md)
- [Local monitoring workflow](docs/monitoring.md)
- [Docker and CI guide](docs/containers-ci.md)
- [SageMaker container contracts](docs/sagemaker-contracts.md)
- [Fixed-parameter training reproduction](docs/training-reproduction.md)
- [MLflow guide](docs/mlflow.md)
- [MLOps roadmap](docs/roadmap.md)
- [AWS batch workflow](docs/aws-batch-workflow.md)

The [Telco MLOps repository](https://github.com/anesriad/Telco-Customer-Churn-ML)
inspires the engineering structure. Our temporal evaluation and weekly scoring
contract are specific to this retail problem.
