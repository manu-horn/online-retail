# Docker and CI

The Dockerfile has separate `training` and `inference` targets. Data and reference
artifacts are mounted at runtime, keeping the frozen model and threshold 0.62
separate from both images. A default build still produces inference. These are
command-line targets. Additional `sagemaker-training` and `sagemaker-inference`
targets adapt the same logic to managed job paths and HTTP serving; see
[SageMaker contracts](sagemaker-contracts.md).

## Build and score

From the repository root on Linux, with Docker running:

```bash
docker build --build-arg VCS_REF="$(git rev-parse HEAD)" --tag online-retail:local .
mkdir -p artifacts/container-output
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nosuid,size=128m \
  --mount "type=bind,source=$PWD/data/interim,target=/input,readonly" \
  --mount "type=bind,source=$PWD/artifacts/bundles/xgboost-v1,target=/bundle,readonly" \
  --mount "type=bind,source=$PWD/artifacts/container-output,target=/output" \
  online-retail:local \
  --purchases /input/purchases.parquet --bundle /bundle \
  --cutoff 2011-06-06 --coverage-end 2011-12-09 --output /output/replay-001
```

Use an existing trusted bundle from the [inference guide](inference.md). Supply a
new output directory for each run. The coverage boundary is an assertion about
complete ingestion, not the last transaction date. Output files belong to the
invoking user; JSON application logs go to stderr.

The image defaults to a non-root user. The command above overrides its UID/GID to
match the writable host directory. Base images and uv are pinned by digest;
`uv sync --locked --no-dev --no-editable` installs the package without MLflow,
Optuna, SHAP or LightGBM. Development installs still include those training tools.
The build context excludes data, artifacts, credentials and Git history.
`package_model.py` is also available by overriding the entrypoint with `python`.

## Build and reproduce training

The training target adds the locked `training` dependency group, the reproduction
entrypoint and source/configuration files for MLflow provenance. It does not include
notebooks, datasets or reference models. From the repository root:

```bash
docker build --target training --build-arg VCS_REF="$(git rev-parse HEAD)" --tag online-retail:training-local .
mkdir -p artifacts/container-training
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nosuid,size=256m \
  --mount "type=bind,source=$PWD/data/processed,target=/input/data,readonly" \
  --mount "type=bind,source=$PWD/artifacts/final_roc_policy,target=/input/reference,readonly" \
  --mount "type=bind,source=$PWD/artifacts/container-training,target=$PWD/artifacts/container-training" \
  online-retail:training-local \
  --data /input/data --reference /input/reference \
  --output "$PWD/artifacts/container-training/xgboost-001" \
  --tracking-dir "$PWD/artifacts/container-training/tracking"
```

Use a new `--output` directory each time. Input mounts are read-only; the output
mount contains both the new reproduction and a separate local MLflow store.
The original host `mlflow.db` is not modified. `--tracking-dir` is also available
for non-container runs; omitting it retains the usual repository-local defaults.

**Keep the tracking mount's absolute path identical on host and container.** Local
MLflow stores absolute artifact URIs. A store created under `/output` in a container
would refer to a path that may not exist on the host. With the command above, inspect
the container-created runs from the host using:

```bash
uv run mlflow ui --backend-store-uri "sqlite:///$PWD/artifacts/container-training/tracking/mlflow.db" --host 127.0.0.1 --port 5001
```

The tracked Git revision uses the build argument when Git is absent in the image.
Git branch and dirty status are `unknown` there; the logged source archive records
the files actually executed. A build from uncommitted files is not identified by
the commit alone. Container image IDs are recorded by the verification harness.
See [fitting boundaries and parity rules](training-reproduction.md).

## Automated checks

`.github/workflows/ci.yml` runs on pull requests, pushes to `main` and `feature/**`,
and manual dispatch. It checks the lockfile, Ruff lint/format and pytest before
building both targets with commit-SHA/target tags. Actions are pinned by commit.
The two CLI container checks run as independent matrix jobs. A separate
`sagemaker-contracts` job builds both adapters and checks the train/archive/serve
sequence using synthetic data. The adapter job passed remotely for commit `96cdfc2`; see
[run 45679024871](https://github.com/manu-horn/online-retail/actions/runs/45679024871).

The inference check runs a synthetic end-to-end fixture:

```bash
uv run --locked python tests/container_smoke.py --image online-retail:local --report artifacts/ci/container-smoke.json
```

It compares host/container probabilities and decisions, exercises packaging,
checks JSON success logs, rejects invalid coverage and existing outputs, and checks
that private inputs and training-only dependencies are absent. Containers run with
read-only input mounts/root filesystem and disabled networking. Synthetic data
means CI needs neither the retail workbook nor saved model artifacts.

The training check uses a tiny chronological fitting population and saved reference:

```bash
uv run --locked python tests/training_container_smoke.py --image online-retail:training-local --report artifacts/ci/training-smoke.json
```

It checks exact host/container prediction parity, reference preservation, output
overwrite rejection, non-root defaults and a host-side reload of the MLflow model.
Its test Parquet is deliberately unreadable, so accidentally loading test rows
fails the check. No retail data, AWS credentials or registry access is needed.

These jobs validate code and transport, not model quality or campaign effectiveness.
There is no image publishing, cloud deployment or CD pipeline yet.

## Verification status

All 54 tests pass locally; Ruff and actionlint pass. Both image targets build locally. A separate inference-only,
non-editable environment successfully packaged and scored the synthetic fixture,
matching the host's saved probabilities, decisions and schema.

The local image now builds successfully, and the synthetic container test passes
with identical host/container probabilities and decisions. Packaging, overwrite
protection, invalid coverage, non-root default user and dependency isolation checks
also pass. The report is saved locally at `artifacts/ci/container-smoke.json`.
The frozen retail model also scored 4,951 customers with 682 contacts for the
June 6 replay; saved scores match the previous host replay exactly, excluding
the new run ID. See `artifacts/ci/retail-container-parity.json`.

On the development machine, Docker and containerd storage were migrated to
`/home/docker-storage/docker` and `/home/docker-storage/containerd`. Both services
are active; the existing four images, three stopped containers and three volumes
remain present. Root free space increased from roughly 662 MB to 4.7 GB, with
about 300 GB still available on `/home` after the image build. Rollback data is
kept outside the repository by the one-off migration script.

The Dockerfile checks for at least 4 GiB free before dependency installation;
allow additional headroom for image export.

[GitHub Actions run 45540424765](https://github.com/manu-horn/online-retail/actions/runs/45540424765)
passed for commit `81a3158`: lockfile, lint, formatting, pytest, image build and
synthetic container checks all succeeded on a GitHub-hosted runner.
No image has been published or deployed.


The training target also passes its synthetic check, including host-side MLflow
model reload (`artifacts/ci/training-smoke.json`). The rebuilt inference target
passes its existing checks (`artifacts/ci/inference-smoke.json`).

A full retail fit inside the training image reproduced all **28,001** holdout
predictions exactly after fitting **146,415** snapshots. Results are in
`artifacts/container-training/xgboost-001/`, with MLflow run
`4b25ec56da7140dd91baef7aefb0816b` in the separate container-training store.
The reference model/policy are unchanged. The parity summary is
`artifacts/ci/training-retail-parity.json`.

The expanded matrix also passes remotely: [run 45665860790](https://github.com/manu-horn/online-retail/actions/runs/45665860790)
for commit `161b858` passed code checks, the inference container check and the
training container check, including host-side MLflow reload. The earlier linked
run covers the initial inference-only workflow.

Smoke-test assertions include subprocess exit codes, stdout and stderr so a
container failure is visible in CI logs rather than only an assertion location.
