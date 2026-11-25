# Docker and CI

The Dockerfile packages the existing batch inference commands. Model artifacts and
purchase data are mounted at runtime, keeping the frozen model and threshold 0.62
separate from the image. This is a command-line inference image; SageMaker's HTTP
serving contract and a training image remain future work.

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

## Automated checks

`.github/workflows/ci.yml` runs on pull requests, pushes to `main` and `feature/**`,
and manual dispatch. It checks the lockfile, Ruff lint/format and pytest before
building an image tagged with the commit SHA. Actions are pinned by commit.

The container job runs a synthetic end-to-end fixture:

```bash
uv run --locked python tests/container_smoke.py --image online-retail:local --report artifacts/ci/container-smoke.json
```

It compares host/container probabilities and decisions, exercises packaging,
checks JSON success logs, rejects invalid coverage and existing outputs, and checks
that private inputs and training-only dependencies are absent. Containers run with
read-only input mounts/root filesystem and disabled networking. Synthetic data
means CI needs neither the retail workbook nor saved model artifacts.

These jobs validate code and transport, not model quality or campaign effectiveness.
There is no image publishing, cloud deployment or CD pipeline yet.

## Verification status

All 40 existing tests pass; Ruff and actionlint pass. A separate inference-only,
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
