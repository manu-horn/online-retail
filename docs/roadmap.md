# MLOps roadmap

## Completed locally

- uv environment, installable package and repository foundation.
- Two-sheet ingestion, source checksum and purchase cleansing with removal reports.
- Sales/customer exploration, sparse-history handling and weekly past-only features.
- Binary 50-day labels, complete follow-up and chronological partitions.
- Training-only RFM diagnostics, normalized scores and descriptive customer segments.
- Processed-feature analysis, redundancy and multicollinearity diagnostics.
- Logistic references and Optuna tree searches with expanding temporal folds.
- Calibration diagnostics without fitted recalibration.
- ROC-AUC model selection, final fitting, later policy holdout and frozen threshold.
- Final test evaluation; local model, predictions, policy and reports saved.
- Local MLflow tracking, saved-artifact imports and executable development training/evaluation scripts.
- Frozen model/policy bundles, local cutoff scoring, JSON application logs and historical inference parity.
- Fixed-parameter final-fit reproduction, holdout prediction parity and MLflow lineage.

See [results](results.md) for the selected XGBoost model and threshold 0.62.
Test has already been evaluated; it is not a fresh selection dataset.

## Next: batch-oriented AWS workflow

Follow the [AWS batch workflow roadmap](aws-batch-workflow.md), one milestone at a time:

1. **Local commands:** build cutoff features, reproduce fixed-parameter training,
   package the model/policy, and score without notebooks.
2. **Versioning and tracking:** manifests and local MLflow connect data, code,
   environment, models, policies and outputs.
3. **Containers and CI:** reproducible training/inference images and automated checks.
4. **First AWS replay:** S3 → SageMaker Training → model in S3 → Batch Transform
   → probabilities and targeting table in S3.
5. **Orchestration:** move features to Processing, then connect working jobs with
   SageMaker Pipelines and add weekly scoring triggers.
6. **Operations:** monitor jobs/data and delayed outcomes; define promotion,
   rollback, retraining and new evaluation periods.

**Current implementation:** local tracking, development training/evaluation scripts,
versioned frozen bundles and batch scoring with structured logs are implemented.
See [MLflow integration](mlflow.md), [local inference](inference.md) and
[training reproduction](training-reproduction.md). Fixed-parameter fitting matches
the saved policy-holdout predictions exactly. A training container remains next.
An inference Dockerfile and GitHub Actions CI workflow are added. Local image build and synthetic container checks pass;
both GitHub Actions jobs also passed for commit `81a3158`.
See [Docker and CI](containers-ci.md). Cloud resources are not required yet.

Training and weekly scoring are separate. Preserve the current evaluated model
and policy while reproducing them. The first cloud run uses selected parameters,
not a fresh hyperparameter search. Test has already been evaluated.

APIs, interactive interfaces and continuously running services are deferred.
Real campaign economics and production acceptance criteria remain open decisions;
controlled campaign experiments are outside current implementation scope.
