# AWS batch workflow roadmap

## Goal and scope

Build a reproducible weekly scoring workflow around the existing customer
repurchase model. Start locally, then demonstrate one small AWS run:
**S3 → SageMaker Training → model in S3 → Batch Transform → predictions in S3**.

This is a plan, not implemented infrastructure. The historical dataset will be
used to replay scoring dates; it does not provide a live transaction feed.
The selected XGBoost model, threshold 0.62 and existing evaluation remain reference
artifacts. Costs remain illustrative. See [modeling](modeling.md) and [results](results.md).

## Target architecture

```mermaid
flowchart LR
    A[Transactions available by cutoff] --> B[Validate and build features]
    B --> C[Versioned snapshots in S3]
    C --> D[Occasional SageMaker training job]
    D --> E[Versioned model bundle in S3]
    C --> F[Weekly Batch Transform]
    E --> F
    F --> G[Purchase probabilities in S3]
    G --> H[Apply versioned policy]
    H --> I[Customer targeting table in S3]
```

Training and weekly inference are separate workflows. Scoring does not retrain.
The policy is separate from probability estimation so its assumptions and threshold
can be versioned without changing the model. Feature generation starts locally;
a later milestone moves it to SageMaker Processing.

## Current status

Local MLflow tracking, artifact imports and development training/evaluation scripts
are implemented, alongside local batch inference, frozen model/policy bundles and
structured scoring logs. Fixed-parameter training reproduction, Docker/CI and AWS
execution remain planned. See [local inference](inference.md).

## Milestones

### 1. Local training and scoring interface — partly complete

Reuse package functions behind commands with explicit input/output paths and
configuration. Notebooks remain exploration and reporting tools.

- Build customer features for a specified cutoff using only earlier transactions.
  Inference needs no future labels; training snapshots still require complete
  50-day outcomes. Validate source coverage at the cutoff.
- Train using fixed selected parameters and explicit fitting populations. First
  reproduce the existing final fit; do not rerun Optuna or select from test.
- Package the fitted preprocessing/model, ordered feature schema, policy and manifest.
- Score a feature table and apply the specified policy separately.

**Current evidence:** the frozen-model scoring replay is implemented and matches
saved probabilities exactly; fixed-parameter final-fit reproduction is still pending.

**Done when:** one local historical replay runs without a notebook; predictions
match the existing model within numerical tolerance; input schema and customer/cutoff
keys are checked; rerunning identical inputs/versions gives equivalent results.
Reproductions write new output locations and preserve the original evaluated bundle.

### 2. Versioning and local experiment tracking

Use manifests with source checksums, code revision, uv lockfile identity, feature
schema, cutoffs, parameters and artifact versions. Local MLflow already links
runs, metrics and artifacts; finish the inference bundle and version contract. Keep credentials and generated data out of Git.

A bundle contains the fitted pipeline, schema, policy configuration and manifest.
The policy records its threshold, costs, model reference and selection population.
MLflow can record the bundle; cloud-hosted tracking is not required for the first run.

**Done when:** a run can be traced from its input data and code to its model,
policy and predictions. The selected bundle is identified explicitly, not by an
unqualified “latest” file.

### 3. Container packaging and automated checks

Create training and inference entrypoints using the existing Python 3.12/uv
environment. Keep prediction dependencies separate from notebook/SHAP tooling
where practical. Adapt serving code to SageMaker's container contract; a pickle
alone is not a Batch Transform deployment. Use ECR for custom AWS container images.

GitHub Actions runs tests, lint and a small inference check before building a
versioned image. Publish images with immutable identifiers such as a commit tag
and record the image digest with each run.

**Done when:** a local container reproduces package-level predictions and the
training/inference contracts pass checks. No cloud jobs are needed yet.

### 4. First AWS training and batch prediction run

Choose one AWS region, a spending limit and supported CPU instance types before
execution. Configure scoped IAM roles, billing alerts and private S3 storage.

1. Upload a versioned feature dataset and fitting configuration to S3.
2. Run one SageMaker Training job with the selected parameters.
3. Save its model bundle and run metadata to S3.
4. Run Batch Transform on a prepared historical scoring table.
5. Apply the versioned policy and save the customer targeting table.
6. Compare local/cloud outputs and inspect CloudWatch job logs.

Suggested S3 prefixes: `datasets/<version>/`, `models/<version>/`,
`policies/<version>/`, `predictions/<cutoff>/<run-id>/`, `reports/<run-id>/`.
Choose actual bucket names and retention rules during implementation.

**Done when:** one end-to-end replay succeeds, records job/image/model/policy
versions, preserves row identity and matches local predictions. Budget alerts
notify; they are not a guaranteed spending cap. Review completed-job resources
and retained storage after the exercise.

### 5. Managed feature generation and orchestration

Move cleansing, cutoff feature construction and evaluation scripts into SageMaker
Processing. Connect the independently working jobs using SageMaker Pipelines.
Start with manual triggers; add weekly scheduling after rerun behaviour is verified.

**Done when:** scoring runs can be repeated safely for a cutoff, failed jobs are
visible, and the workflow does not silently overwrite approved outputs. Keep
training triggers separate from the weekly scoring schedule.

### 6. Monitoring and release management

Track source coverage, missingness, feature/score distributions, contact rate and
job failures. Once 50-day outcomes mature, report ranking, calibration and policy
performance by week and customer-history group using our evaluation helpers.

Define model/policy promotion, rollback, retraining triggers and new evaluation
periods. Historical test has already been inspected; do not repeatedly use it for
selection and call it a fresh final evaluation. Weekly snapshots are correlated.

**Done when:** an approved bundle can be identified and rolled back, monitoring
outputs are reproducible, and owners/response rules are documented.

## Output contract

Each scored row records customer ID, cutoff, purchase probability, contact decision,
model version, policy version, feature/data version and scoring run ID. Preserve
customer/cutoff identity through Batch Transform; verify row counts and joins.
Targets and future outcomes are not inference inputs. Invalid or incomplete data
should produce a visible failure rather than silently changing the population.

## Deferred work

FastAPI, Gradio, ALB and a continuously running ECS service are optional later
extensions if interactive scoring is needed. A feature store, distributed training,
GPU workloads and a managed MLflow server are not first-milestone requirements.
Controlled campaign experiments are outside the current implementation scope.

## References

- [Telco reference repository](https://github.com/anesriad/Telco-Customer-Churn-ML): modular code, MLflow, containers and CI/deployment ideas.
- [AWS beginner guide](https://www.marinawyss.com/blog/aws-for-machine-learning-beginners): start with a small storage/training/batch-inference project.
- [SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html).
- [Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html).
- [SageMaker Processing](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html).
- [Custom inference containers](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html).
- [SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html).
