# Online Retail: customer repurchase prediction

Predict an identified customer’s next-purchase range (days **0–20**, **21–49**, or
**no purchase within 50 days**) at weekly cutoffs using
historical transactions from [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
See the current prediction contract for eligibility and temporal evaluation. This repository will grow into
an end-to-end MLOps workflow, one milestone at a time.

Read the [prediction contract](docs/problem-definition.md) and
[roadmap](docs/roadmap.md) before modeling. There is no trained model yet.

## Environment

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). `pyproject.toml` declares
dependencies; commit `uv.lock` to retain the resolved versions.

```bash
uv sync --locked --group notebooks
uv run --locked python -c "import online_retail; print(online_retail.__doc__)"
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked jupyter lab
```

Select the project's `.venv` Python kernel when opening notebooks. Development
tools install by default; the `notebooks` group is optional. Use `uv add` to add
runtime dependencies and `uv add --group dev` for development tools. Use
`uv lock` when intentionally changing dependency resolution.

## Layout

```text
configs/prediction.toml      weekly snapshot and chronological partition dates
data/README.md              source attribution and local data conventions
data/raw/                   original workbook, ignored by Git
docs/problem-definition.md  target, cohort, leakage rules, evaluation
docs/roadmap.md             staged MLOps work
notebooks/01_eda.ipynb       existing exploratory notebook
src/online_retail/          installable Python package
pyproject.toml              dependencies and tooling configuration
uv.lock                    exact dependency resolution
```

Add pipeline modules and `tests/` with the first executable data pipeline.
Keep reusable logic in the package and exploration in notebooks. Keep raw data,
trained models, credentials, and experiment outputs out of Git. Clear notebook
outputs and inspect notebooks for embedded data or secrets before committing.

## Initial data cleansing

```bash
uv run --locked python -m online_retail.data.preprocess
uv run --locked pytest -q
```

The loader reads both Excel sheets; preprocessing writes identified purchase
lines to `data/interim/purchases.parquet` and sequential removal counts to
`data/interim/purchases.report.json`. See [cleansing rules](data/README.md).
Repeated lines with matching non-quantity fields are consolidated by summing quantity. Country encoding and customer features come later.

## Initial commit

Once the environment checks pass, initialize Git if necessary, inspect staged
changes, and create the foundation commit:

```bash
git init -b main
git add .gitignore .python-version pyproject.toml uv.lock README.md configs docs src data/README.md
git diff --cached --stat
git diff --cached
git commit -m "chore: bootstrap retail repurchase project"
```

The existing notebook is intentionally left out of this explicit first staging
command until its contents and outputs have been reviewed.
