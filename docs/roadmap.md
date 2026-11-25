# Simple MLOps roadmap

The project demonstrates one reproducible batch workflow:

**Data preparation → training → MLflow tracking → saved model → batch predictions → basic monitoring**

The AWS extension is one small historical replay, following the original plan:
**S3 → SageMaker Training → model in S3 → Batch Transform → predictions in S3**.
This is a learning project using a static dataset, not an unattended production service.

## Tools and current status

| Part | Tools | Status |
|---|---|---|
| Data preparation and customer features | Python, pandas, Parquet, uv | Implemented; notebooks explore data and package functions build features |
| Modeling and temporal validation | scikit-learn, XGBoost, Optuna | Completed; selected model and threshold are frozen |
| Experiment/model tracking | MLflow, SQLite | Implemented locally |
| Training reproduction and batch scoring | Python commands, versioned model/policy bundles | Implemented and checked against saved predictions |
| Reproducible environments | Docker, uv lockfile | Training and inference images verified locally |
| Continuous integration | GitHub Actions, pytest, Ruff | Code and container checks implemented |
| Basic monitoring | pandas, NumPy, matplotlib, scikit-learn; optional MLflow import | Reports/logs implemented; performance requires complete 50-day follow-up |
| AWS replay | S3, SageMaker Training, Batch Transform; supporting IAM/ECR and job logs | Adapters verified locally; cloud execution deferred |

Commands and details: [local inference](inference.md), [training reproduction](training-reproduction.md),
[monitoring](monitoring.md), [Docker/CI](containers-ci.md), [SageMaker contracts](sagemaker-contracts.md).

## What remains for this scope

1. Review the final commands/documentation and verify CI for the completed branch.
   Keep a short reproducible demo of training, scoring and monitoring.
2. Resolve AWS account access and verify whether the proposed replay can fit the
   preferred **$0 out-of-pocket budget**. The selected profile is
   `purchase-propensity`, region `us-east-2`; the user confirmed the account is closed.
   Recovery/credit eligibility is unresolved.
3. If access and costs permit, configure the necessary IAM permissions, private S3
   storage and ECR images; run one SageMaker training job and one Batch Transform
   job. Compare their outputs with the verified local replay and clean up resources.

The local workflow can be demonstrated while AWS is unavailable. Do not describe
local container checks as cloud deployment. See the [small AWS replay plan](aws-batch-workflow.md).
No cloud resources have been created by this workflow.

## Monitoring is sufficient for the initial project

Keep the implemented per-batch reports, structured failure logs, missingness/drift
checks, contact rates and delayed performance by customer-history group. Initial
warning limits are review heuristics, not validated production alarms. Expected
cohort aging can produce warnings; it does not by itself establish model degradation.

The dataset is a historical replay. Future model changes need an appropriate new
evaluation plan; the existing test period is not a fresh selection dataset.

## Deferred, not required to finish

Scheduling, dashboards/weekly trend indexes, notifications, MLflow registry
promotion, automated rollback, GHCR publishing, automatic retraining and managed
orchestration are optional later extensions. We are not adding them as milestones.

An interactive API/UI, additional model families and real campaign experiments
are also outside this batch demonstration. The Flask/Gunicorn adapter exists for
Batch Transform's HTTP contract. Campaign costs remain illustrative.
