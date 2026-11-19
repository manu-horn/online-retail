# Results and saved model

This summarizes the completed experiment, not a deployed marketing system.
The prediction contract is in [problem-definition.md](problem-definition.md);
training, calibration and policy mechanics are in [modeling.md](modeling.md).
Numbers come from the local saved reports under `artifacts/`.

## Development results

The original log-loss experiments produced these external-validation results:

| Model | Log loss ↓ | Brier ↓ |
|---|---:|---:|
| Prior reference | 0.6001 | 0.2041 |
| L2 logistic | 0.6769 | 0.2294 |
| Elastic-net logistic | 0.6618 | 0.2271 |
| LightGBM | 0.5642 | 0.1888 |
| XGBoost | 0.5583 | 0.1861 |
| Random Forest | 0.5510 | 0.1826 |

The current experiment instead selects by internal mean weekly ROC-AUC:

| Model | Best CV ROC-AUC ↑ |
|---|---:|
| XGBoost | **0.74444** |
| Random Forest | 0.74425 |
| LightGBM | 0.74412 |

XGBoost wins by a small margin, not evidence of a meaningful superiority across
all populations. These are different searches with different selection criteria;
the earlier validation winner does not determine the final model.

## Frozen test results

XGBoost was refitted on 146,415 mature pre-policy snapshots. Development policy
data selected threshold 0.62. The same fitted model and threshold were then used
on 104,034 test customer-week snapshots.

| Threshold-agnostic metric | Test value |
|---|---:|
| ROC-AUC | 0.7948 |
| Average precision | 0.6504 |
| Log loss | 0.5098 |
| Mean weekly log loss | 0.5088 |
| Binary Brier score | 0.1673 |

At the frozen threshold 0.62:

| Metric | Test value |
|---|---:|
| Buyer precision | 0.7077 |
| Buyer recall | 0.4224 |
| Buyer F1 | 0.5291 |
| Accuracy | 0.7962 |
| Contacted customer-week instances | 16,826 |

| Actual outcome | No contact | Contact |
|---|---:|---:|
| No purchase | 70,929 | 4,919 |
| Purchase | 16,279 | 11,907 |

With the illustrative costs documented in modeling.md:

| Policy | Mean assumed cost (£ / customer-week) |
|---|---:|
| Frozen model threshold | 5.7657 |
| Contact nobody | 6.7733 |
| Contact everybody | 13.6453 |

See the [worked cost example](modeling.md#worked-cost-example-threshold-062) for
the confusion-matrix calculation and the separation of spending and penalties.

Assumed total cost is £599,830, about 14.9% below the no-contact reference. This is
not observed marketing expenditure or measured profit. Rows are repeated weekly
decisions, not unique customers. Predicting purchases does not tell us whether
contact causes extra purchases. The threshold trades recall for precision: many
buyers remain uncontacted under these assumed costs.

## Saved artifacts

All paths below are relative to the repository root and are Git-ignored.

| Path | Contents |
|---|---|
| `artifacts/final_roc_policy/model.pkl` | Final XGBoost pipeline, including fitted imputation and missing indicators |
| `artifacts/final_roc_policy/frozen_policy.json` | Feature order, parameters, costs, fit/policy dates and threshold |
| `artifacts/final_roc_policy/report.json` | Frozen settings, probability/0.5 diagnostics and selected-policy test results |
| `artifacts/final_roc_policy/policy_predictions.parquet` | Development policy probabilities and outcomes |
| `artifacts/final_roc_policy/policy_cost_curve.csv` | Policy-holdout threshold costs, including contact-nobody option |
| `artifacts/final_roc_policy/test_predictions.parquet` | Test probabilities, outcomes and frozen contact actions |
| `artifacts/roc_auc_models/<model>/` | Trial table, all fold metrics, selected parameters, validation results and model |
| `artifacts/model_comparison.csv` | Earlier log-loss experiment's validation comparison |

The classifier pipeline does **not** include transaction cleansing, customer
snapshot construction or the threshold. `frozen_policy.json` supplies the ordered
22 input features and the separate decision rule. Raw customer IDs or transactions
cannot be passed directly to this classifier.

Example using a trusted local artifact and an already engineered snapshot table:

```python
import json
import pickle
from pathlib import Path

folder = Path("artifacts/final_roc_policy")
policy = json.loads((folder / "frozen_policy.json").read_text())
with (folder / "model.pkl").open("rb") as handle:
    model = pickle.load(handle)

# snapshots uses the same past-only feature definitions as training.
probability = model.predict_proba(snapshots[policy["feature_columns"]])[:, 1]
contact = probability >= policy["threshold"]  # Current saved threshold: 0.62.
```

Only load trusted pickle files and use the locked project environment. Artifacts
are local, not pushed or registered in a model registry. Source control and
`uv.lock` alone do not preserve these generated files.

## Reproduction and next work

Run notebooks 01–04 in order with the project kernel; notebook 02 rebuilds feature
tables and notebook 03 describes training data. Notebook 04 retains earlier
experiments and the current ROC-AUC section. Existing final reports are loaded
rather than refitting or reselecting the policy on rerun. Earlier experiment cells
may still retrain their models; the ROC-AUC section has a separate rerun switch.
`run_final` refuses to overwrite an existing final report.

There is no production scoring service, experiment registry, deployment or
monitoring system yet. Next work is artifact/data versioning and reproducible
batch inference, followed by experiment tracking and operational monitoring.
Business costs, repeated-contact rules and performance acceptance criteria need
real requirements before deployment. See [roadmap](roadmap.md).

## Additional F1 policy comparison (exploratory)

The policy holdout selects threshold 0.48 by buyer F1 alone (holdout F1 0.5701).
Costs are computed only afterward. The fitted XGBoost model is unchanged.

| Policy | Threshold | Test precision | Test recall | Test F1 | Mean assumed test cost (£) |
|---|---:|---:|---:|---:|---:|
| Default | 0.50 | 0.5514 | 0.6153 | 0.5816 | 6.3070 |
| F1-selected | 0.48 | 0.5291 | 0.6453 | 0.5815 | 6.4845 |
| Cost-selected | 0.62 | 0.7077 | 0.4224 | 0.5291 | 5.7657 |

The small development F1 gain does not transfer to test. The F1 threshold contacts
more customers and has higher assumed costs. No threshold was selected on test,
but the test comparison is exploratory reuse, not a new untouched evaluation.
Artifacts are under artifacts/f1_policy/; notebook 04 shows confusion matrices and
threshold curves. The original model.pkl and frozen_policy.json remain unchanged.

## Frozen-model diagnostic evaluation

Notebook [05_evaluation.ipynb](../notebooks/05_evaluation.ipynb) verifies saved
probabilities against the frozen pipeline, joins outcomes/features one-to-one, and
reports overall ROC/PR/reliability curves, confusion matrix and assumed costs.
It does not train, recalibrate or select thresholds. Weekly and customer-history
slices are descriptive reuse of test, not new independent selection results.

The fixed 0.62 policy contacts no one-invoice snapshots (precision undefined,
recall zero). Buyer recall is 1.8% for 2–3 prior invoices and 60.2% for 4+ invoices.
Unseen-in-final-fit IDs have ROC-AUC 0.733 and recall 18.7%, versus 0.821 and 47.5%
for seen IDs. Unseen customers still have past transactions at scoring time.
Groups use cutoff-specific history, not full-dataset customer totals. People may
move between invoice-count groups, so their distinct-customer counts overlap.

Weekly ROC-AUC ranges 0.764–0.819; weeks share customers and overlapping labels,
so these are descriptive ranges, not independent uncertainty estimates. Tables
include sample sizes, buyers, purchase prevalence, contact rates and constant-action
cost references. Smaller slice ROC-AUC/AP values and precision/recall can be
undefined; the notebook explains missing values. F1 uses zero for undefined cases.

Tables are under artifacts/evaluation/: overall.csv, weekly_metrics.csv,
history_metrics.csv and identity_metrics.csv. Reusable code lives in
src/online_retail/modeling/evaluation.py. No controlled campaign experiment is
included in this notebook.

## SHAP explanations by customer history

Notebook 05 uses the standard SHAP library's TreeExplainer and beeswarm/waterfall
plots on a uniform random sample of 5,000 test snapshots (seed 42). The saved
imputer transforms inputs; missing indicators remain separate named features.
Tree-path-dependent explanations need no new background sample. Contributions
plus baseline are checked against margins and frozen probabilities. Units are
purchase log-odds, not probability percentage points.

There is one standard beeswarm per invoice-history group and one waterfall example
per group. Examples are selected nearest the group's median predicted probability,
without using actual outcomes; they are illustrations, not representative estimates.
Their table records ID, cutoff, actual outcome, probability and contact decision.
Groups show sample denominators. This is snapshot-weighted, not equal weight per
unique customer. No refitting, feature selection or threshold changes occur.
Correlated predictors share attribution; SHAP describes model predictions, not
causality or guaranteed predictive usefulness. Different group input distributions
can yield different contributions.

The 2–3 invoice group has the weakest ROC-AUC (0.651 versus 0.711 for one invoice
and 0.767 for 4+ invoices). At 0.62, its 144 contacts include 94 TP and 50 FP;
5,230 buyers are missed. Precision is 65.3%, recall 1.8%, F1 0.0344. Accuracy 81.8%
compares with 81.6% from always predicting non-purchase. One-invoice recall is
still worse (zero); no single group is worst on every metric. AP and loss also
depend on the different outcome prevalences.

Reusable code: src/online_retail/modeling/explanations.py. Outputs under
artifacts/evaluation/shap/ are global_importance.csv, history_importance.csv,
sample_contributions.parquet and examples.csv. Figures are saved in notebook 05.
The standard shap dependency is declared in pyproject.toml and locked in uv.lock.
