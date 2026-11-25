# MLOps roadmap and tools

The immediate goal is a reproducible local batch workflow, with optional AWS
execution once account access and costs allow it. Preferred cloud spend is **$0**.
The user confirmed that the configured AWS account is closed; support/account
recovery is pending. No cloud resources have been created by this workflow.

## Already working

- Data provenance/cleansing and past-only weekly customer features: pandas, Parquet, Python.
- Chronological tuning, frozen XGBoost/threshold and evaluation: scikit-learn, XGBoost, Optuna.
- Experiment tracking and reproducible fixed-parameter fitting: MLflow and SQLite.
- Versioned bundles, scoring and structured events: Python, SHA-256 manifests, JSON logs.
- CLI and SageMaker-compatible training/inference images: Docker, uv, Flask/Gunicorn.
- CI for code and container contracts: pytest, Ruff, GitHub Actions.
- Initial local monitoring: fixed reference, drift/missingness, prediction/contact
  distributions, history groups, JSON/Markdown/PNG reports and optional MLflow import.
- Separate mature-outcome monitoring: scikit-learn metrics, saved scoring cohorts,
  50-day coverage gate and historical-backfill detection.

See [monitoring commands](monitoring.md), [training reproduction](training-reproduction.md),
[Docker/CI](containers-ci.md) and [SageMaker contracts](sagemaker-contracts.md).
The evaluated model remains frozen. The historical test has already been inspected.

## Remaining local work, in order

| Step | Work still needed | Tools | Done when |
|---|---|---|---|
| 1 | Review/tune monitoring limits using development periods; document owner and response for each warning | Current Python monitoring reports, `configs/monitoring.toml` | Review rules are justified; expected cohort aging is distinguished from input failures |
| 2 | Combine batch statuses and delayed outcomes into a weekly trend/run index | pandas, Parquet/CSV, matplotlib, existing MLflow UI | One view shows successes/failures, drift, contact rates, matured metrics and overdue outcomes |
| 3 | Make ingestion coverage/backfill handling explicit and run feature dataset generation without notebooks | Existing preprocessing/features modules, Python CLI, checksums/manifests | A new data delivery has trusted coverage metadata and reproducible training/scoring inputs |
| 4 | Schedule weekly scoring and due outcome checks, with retry/deduplication rules | Local cron initially, Python commands, JSON logs | A cutoff/version has an identifiable published result; retries do not duplicate or overwrite it |
| 5 | Add operational notifications and a response runbook | Scheduler exit status, JSON logs; notification destination to be chosen | Job failures and actionable warnings reach an owner; recovery steps are tested |
| 6 | Register releases, define promotion criteria and test rollback | MLflow Model Registry aliases, bundle/policy/reference manifests, Git | A release pins all artifacts; a previous approved release can be restored and replayed |
| 7 | Automate image release/delivery after passing checks | Docker, GitHub Actions, GitHub Container Registry (GHCR) | A tagged release publishes code-only images by digest; promotion remains separate |
| 8 | Back up artifacts/tracking and define retention/restore procedures | Filesystem archives, SQLite backups, manifests | A trusted model, policy, reference and tracking history can be restored on another machine |
| 9 | Define retraining triggers and the next validation population | Existing Python/scikit-learn temporal evaluation; Optuna only if justified | Drift prompts investigation; a candidate is evaluated on an appropriate new forward period before promotion |
| 10 | Refresh the README/demo and verify fresh-clone instructions | Markdown, CLI commands, GitHub Actions | A reader can reproduce the local lifecycle and understand every implemented/planned boundary |

Steps 1–2 are the recommended next work. The initial monitoring implementation is
complete, but operational thresholds and unattended scheduling are not validated.
For the static dataset, scheduling must advance explicit historical replay cutoffs;
it must not score today using a feed that ends in 2011. A real recurring deployment
needs a real ingestion source and trustworthy coverage metadata.
Repeated historical test reuse is not a fresh final evaluation. Truly new model
quality claims require suitable new/unseen data; the UCI dataset is a static replay.

GHCR is the proposed no-AWS image delivery destination. GitHub currently documents
public packages as free and container-registry storage/bandwidth as free; review
current terms before publishing. Images contain code/dependencies, not datasets,
credentials or trained models. Publishing is **not implemented or performed yet**.
[GitHub billing](https://docs.github.com/en/billing/concepts/product-billing/github-packages).

MLflow tracking is already present; registry aliases, release approval and rollback
automation are not. [MLflow registry concepts](https://mlflow.org/docs/latest/ml/model-registry/).
No additional orchestration or monitoring framework is required for the first local
workflow: it uses the existing Python environment and generates portable artifacts.

## Conditional AWS work

This remains optional while the account is inaccessible or a zero-charge replay
cannot be established. The selected profile is `purchase-propensity`, region
`us-east-2`; eligibility/remaining credits have not been verified.

| Order | Remaining work | Planned tools |
|---|---|---|
| A | Resolve account closure, verify eligibility/credits, choose instance limits and retention | AWS Settings/Basic Support, read-only AWS CLI, billing information |
| B | Configure scoped permissions, private data/artifact storage and image repositories | IAM, S3, ECR; configuration through AWS CLI/SDK |
| C | Run one fixed historical training job and Batch Transform replay; verify parity and cleanup | SageMaker Training, Batch Transform, current container adapters, CloudWatch logs |
| D | Automate the working steps and weekly triggers | SageMaker Processing/Pipelines and EventBridge, after the manual replay works |
| E | Route cloud failure/data-quality notifications and test promotion/rollback | Our monitoring reports, CloudWatch/SNS, versioned artifacts and MLflow release metadata |

Budget alerts are not a guarantee of zero charges. No cloud jobs or resource
creation should be inferred from the presence of adapters or this plan. See the
[AWS batch workflow](aws-batch-workflow.md) for the proposed architecture.
We retain custom Python monitoring; AWS states that SageMaker Model Monitor is
no longer open to new customers. [AWS availability notice](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html).

## Optional extensions, outside the core batch milestone

- Interactive customer-scoring API or UI if a user needs it; choose FastAPI or a UI
  framework only when its requirements are defined. The Flask adapter serves the
  Batch Transform contract, not a complete end-user application.
- Business cost validation and controlled campaign experiments before claiming
  incremental marketing benefit. Existing campaign penalties are illustrative.
- More product features or model families only when a new evaluation plan supports
  the experiment; adding them is not required to complete the engineering workflow.
