# SageMaker container contracts

This adds local-testable adapters for one CPU training instance and Batch Transform.
It does not create AWS resources or change the selected model, threshold or test
results. Existing `training` and `inference` CLI targets remain available; the
Dockerfile still defaults to CLI inference.

## Training adapter

Build with `docker build --target sagemaker-training -t online-retail:sm-training-local .`.
SageMaker supplies the `train` argument to the image entrypoint. Our adapter uses:

| Path inside the container | Contents |
|---|---|
| `/opt/ml/input/data/training/` | `refit.parquet` and processed `manifest.json` |
| `/opt/ml/input/data/reference/` | Trusted frozen pipeline, policy and saved policy-holdout predictions |
| `/opt/ml/input/config/` | Optional SageMaker channel/resource/hyperparameter JSON configuration |
| `/opt/ml/model/` | Deployable `model.pkl`, `frozen_policy.json`, `manifest.json` |
| `/opt/ml/output/data/` | Reproduction report, holdout predictions, bundle copy and job summary |
| `/opt/ml/output/failure` | Short failure description when training fails |

The existing fixed-fit function repeats the original fitting population and checks
policy-holdout parity before packaging a model. It does not use test rows, Optuna
or a new threshold. Hyperparameter overrides, multiple training hosts and non-File
input modes are rejected. Destinations must be fresh. SIGTERM produces a failed
run rather than a promoted partial fit; checkpoint/resume is not implemented.

SageMaker packages the contents of `/opt/ml/model` into `model.tar.gz`. Training
reports remain separate from the deployable bundle. No container-local MLflow
SQLite database is created here: absolute job-local artifact paths would not make
a portable cloud tracking store. This adapter uses the core fixed-fit dependencies without the MLflow/Optuna
training group. The local CLI workflow retains MLflow; job outputs
can be imported using `scripts/track_results.py` after download, or a remote tracking
service can be added later. Logs are JSON and go to stderr.

## Batch Transform adapter

Build with `docker build --target sagemaker-inference -t online-retail:sm-inference-local .`.
The `serve` argument starts Gunicorn/Flask on port 8080. It loads the checked bundle
from read-only `/opt/ml/model` before accepting requests.

| Endpoint | Behavior |
|---|---|
| `GET` or `POST /ping` | Empty HTTP 200 after successful model loading |
| `GET /execution-parameters` | MultiRecord, 6 MiB payload, one concurrent transform |
| `POST /invocations` | Validate and score one or more JSON Lines snapshot records |

Use `ContentType=application/jsonlines`, `Accept=application/jsonlines`,
`SplitType=Line`, `BatchStrategy=MultiRecord`, `AssembleWith=Line`,
`MaxConcurrentTransforms=1`, and `MaxPayloadInMB=6` when creating a future transform
job. No transform job is submitted by this repository yet. Missing gap values are
JSON `null` and remain missing for the fitted imputer.

Each input record has **exactly** the bundle's numeric feature columns, plus:

- `Customer ID` and `cutoff`: the customer/snapshot key.
- `feature_version`: must match the bundle's feature-definition identity.
- `data_version`: SHA-256 of the source purchase file.
- `coverage_end`: declared exclusive ingestion boundary, at or after the cutoff.

Only eligible Monday cutoffs after policy availability are accepted. Targets and
unexpected fields are rejected. Feature values must be finite numbers or null.
Responses preserve row order and customer/cutoff keys, and return purchase
probability, contact decision, artifact/data versions and a worker execution ID.
The decision still uses threshold **0.62**. Retries or worker restarts may change
the execution ID; the customer/cutoff key is the logical row identity.

Schema/version checks do not prove that supplied features match real transactions.
Use the trusted preparation command with complete observed history. It excludes
future purchases and preserves the same feature definitions as local scoring:

```bash
uv run --locked python scripts/prepare_batch.py --bundle artifacts/bundles/xgboost-v1 --cutoff 2011-06-06 --coverage-end 2011-12-09 --output artifacts/sagemaker/batch-2011-06-06
```

This writes `features.jsonl` and a provenance manifest in a new directory. Upload
only `features.jsonl` as transform input; the manifest is a separate sidecar.
The exporter checks uniqueness across the full file; the HTTP adapter checks it
within each request. Downstream processing must check output counts and uniqueness
across all response files, since Batch Transform can split work and retry requests.

## Local contract check

After building both targets, run:

```bash
uv run --locked python tests/sagemaker_container_smoke.py --training-image online-retail:sm-training-local --inference-image online-retail:sm-inference-local --report artifacts/ci/sagemaker-contracts.json
```

The synthetic fixture runs training with SageMaker-style mounts, creates/extracts
`model.tar.gz`, starts `serve`, sends single/multiple snapshot rows, checks prediction
parity and invalid requests, and verifies graceful shutdown. Inputs and serving
model mounts are read-only. Containers have no network access except their own
loopback; HTTP tests run through `docker exec`. No ports are published.

SageMaker targets default to root to follow AWS's managed-container contract.
CLI targets retain their non-root defaults. Local checks override the UID/GID so
mounted outputs belong to the invoking user. This is a container-contract check,
not an actual SageMaker Local Mode or cloud integration run.

GitHub Actions includes a separate `sagemaker-contracts` job using this synthetic
fixture. AWS execution, ECR publishing, IAM, S3 and cloud-specific permissions
still need a separately configured and verified first cloud replay.

## Verification evidence

All **54 tests** pass locally, along with Ruff and workflow validation. Both
SageMaker targets build, and the synthetic archive/HTTP/shutdown check passes;
its report is `artifacts/ci/sagemaker-contracts.json`.

The full retail fit used **146,415 snapshots** and matched all **28,001** saved
policy-holdout probabilities exactly. Job artifacts are under
`artifacts/sagemaker/retail-job-001/`. The June 6 HTTP replay scored **4,951 customers**
with **682 contacts**, matching the existing local customer/cutoff keys and decisions.
Maximum probability difference was about `5e-11` from JSON serialization, below the
fixed `1e-7` tolerance. Evidence is in that job directory's `http-replay/report.json`.
The final minimal training image was also verified in `retail-job-002`; it
produced a byte-identical bundle to the one used for that HTTP replay.
Original evaluated artifacts remain unchanged. No test labels were read, no policy
was reselected and no AWS resources were created. The new CI job has not run remotely.

## Sources

- [AWS training entrypoint](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html).
- [AWS training storage paths](https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage-env-var-summary.html).
- [AWS model loading and container permissions](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html).
- [AWS Batch Transform HTTP contract](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-batch-code.html).
