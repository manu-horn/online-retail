# Retail repurchase MLOps demo

Run one historical batch to show how the selected XGBoost model goes from a
reproducible fit to predictions and monitoring. The business question is whether
a customer will purchase within 50 days. This example scores customers as of
June 6, 2011, using the frozen contact threshold of 0.62.

## Run the demo

The command uses cleaned purchases, processed snapshots and the trusted frozen
model, policy and policy-holdout predictions. If starting from a fresh clone,
generate those inputs using the [README setup instructions](../README.md#reproduce-from-a-fresh-clone).
The data, model and generated reports are stored locally. The demo expects:

- `data/interim/purchases.parquet` from preprocessing.
- `data/processed/` with `manifest.json`, `train.parquet` and `refit.parquet` from notebook 02.
- `artifacts/final_roc_policy/` with `model.pkl`, `frozen_policy.json` and
  `policy_predictions.parquet` from notebook 04.

From the repository root, choose a new output directory:

```bash
uv run --locked python scripts/demo.py --output artifacts/demo-2011-06-06
```

The command refits with the saved parameters, checks the refit against saved
policy-holdout probabilities and packages the resulting model and policy. It
then builds customer features from purchases before the cutoff, scores them,
compares the batch against the fixed development reference and evaluates
outcomes after the complete follow-up window. It writes a summary, a JSON event
log and a dedicated SQLite MLflow store for the run.

## Show the workflow

1. Start with `summary.md` in the output directory. It links to the drift chart,
   scoring report and outcome report, and lists the three MLflow run IDs.
2. Open `batch/monitoring.md` and `batch/monitoring.png` to show the reference
   comparison, contact rate and data drift warnings. Longer observed histories
   can trigger a warning as customers age between April and June.
3. Open `outcomes/report.md` to show what happened over the following 50 days,
   including precision and recall by customer history group. The report is for
   one replay week. The [model results](results.md) cover the full historical
   test period.
4. Open MLflow to show the tracked fit, monitoring and outcome metrics:

   ```bash
   uv run --locked mlflow ui --backend-store-uri "sqlite:///$PWD/artifacts/demo-2011-06-06/mlflow.db" --host 127.0.0.1 --port 5000
   ```

   Open [MLflow on port 5000](http://127.0.0.1:5000) and select
   `retail-batch-demo`. Use another port if 5000 is already occupied.

For the engineering part of the demo, show the [Docker and CI guide](containers-ci.md)
and the [passing merged workflow](https://github.com/manu-horn/online-retail/actions/runs/45695629229).
The [AWS plan](aws-batch-workflow.md) remains an optional extension, subject to
account access and the $0 spending preference.

## Share the dashboard

```bash
uv run --locked python scripts/demo_dashboard.py artifacts/demo-2011-06-06 --output artifacts/demo-2011-06-06/dashboard.html
```

On Linux, open it from the repository root:

```bash
xdg-open artifacts/demo-2011-06-06/dashboard.html
```

You can also double-click `dashboard.html` in your file manager. It bundles the monitoring chart, contact rate
and outcome metrics by customer-history group. It contains aggregate results
and run metadata and works offline as a single file. Choose a fresh output
filename if exporting again. The [README preview](../README.md#demo-at-a-glance)
shows the aggregate monitoring chart from the verified June 6 replay.
