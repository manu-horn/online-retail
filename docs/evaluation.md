# Notebook 05: model evaluation

[05_evaluation.ipynb](../notebooks/05_evaluation.ipynb) evaluates the saved XGBoost
pipeline and frozen threshold **0.62**. It does not retrain, recalibrate, select
features or change thresholds. See [modeling](modeling.md) for how these were chosen.

## Population and workflow

All results use **104,034 test customer-week snapshots**, covering **5,558 customers**
and Monday cutoffs June 6–October 17, 2011. A customer may appear in several weeks.
The target is purchase in `[cutoff, cutoff + 50 days)`.

1. Verify saved predictions against the fitted pipeline, matching test rows and
   outcomes one-to-one on customer/cutoff.
2. Report overall ranking, probability quality, classification, confusion matrix
   and assumed costs. Show ROC, PR and reliability curves.
3. Compare performance by week, including observation counts and purchase rates.
4. Compare invoice-history groups and customers seen/unseen in the actual final fit.
5. Explain predictions with standard SHAP beeswarm and waterfall plots.

## Reading the metrics

- **ROC-AUC / average precision:** ranking across thresholds. AP summarizes PR
  behaviour, not trapezoidal PR-AUC; compare it with purchase prevalence.
- **Log loss / Brier:** probability quality; lower is better.
- **Precision / recall / F1:** contact decisions at the frozen threshold.
- **Contact rate / cost:** operational consequences. Costs are hypothetical
  TN/FP/FN/TP values of £0/£15/£25/£10 per snapshot, not measured campaign profit.
  See the [worked example](modeling.md#worked-cost-example-threshold-062).

Reliability plots compare predicted probabilities with observed purchase fractions
in ten quantile bins. They are diagnostics; no calibrator is fitted.

History groups use each snapshot's past invoice count: 1, 2–3, or 4+. Customers can
move between groups, so distinct-customer totals overlap. Unseen IDs are absent
from the final fitting population but have history available when scored.
Undefined ROC-AUC, AP, precision or recall appear as missing where applicable;
undefined F1 uses zero. Compare group sizes and prevalence before interpreting scores.

## SHAP

TreeExplainer uses a uniform sample of **5,000 test snapshots**, seed 42, after the
saved imputer. Missing indicators remain separate explained inputs. Tree-path-dependent
explanations use fitted-tree information without a new background sample.
Contributions are purchase **log-odds**, and additivity/probability checks verify them.

Each group gets a beeswarm and one waterfall. The example is nearest that group's
median predicted probability, chosen without its observed outcome. It illustrates
one prediction, not the whole group. Beeswarm color shows imputed feature value;
position shows contribution. Correlated predictors may share attribution. SHAP
explains the model, not causal effects or a feature's unique predictive value.

## Main conclusions

Overall ROC-AUC is **0.795**, with buyer precision **70.8%** and recall **42.2%**.
The model is selective and performs unevenly across customer histories:

| Prior invoices | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|
| 1 | 0.711 | Undefined: no contacts | 0% |
| 2–3 | 0.651 | 65.3% | 1.8% |
| 4+ | 0.767 | 70.8% | 60.2% |

The middle group has the weakest ranking, but one-invoice buyer recall is worse.
Middle-group accuracy of 81.8% barely exceeds its no-purchase baseline of 81.6%.
Seen/unseen customers have ROC-AUC **0.821/0.733** and recall **47.5%/18.7%**.
These differences describe this fixed model and policy, not intrinsic customer
predictability. Detailed results are in [results.md](results.md).

## Code and outputs

Reusable code: `src/online_retail/modeling/evaluation.py` and `explanations.py`.
Run notebook 05 after creating the final model artifacts in notebook 04; use the
project's uv environment and notebook kernel.

`artifacts/evaluation/` contains overall, weekly, history and identity CSV tables.
Its `shap/` folder contains importance tables, sampled contributions and example
identifiers. Figures are saved as notebook outputs. Generated artifacts are Git-ignored; these guides are version-controlled.

Weekly rows share customers and outcome windows, so they are correlated. Test has
already been inspected: these are descriptive diagnostics, not a fresh selection
experiment. No campaign experiment is included. Changes motivated by these results
need development work and a new evaluation plan.
