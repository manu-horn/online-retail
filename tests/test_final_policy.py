import numpy as np
import pandas as pd

from online_retail.modeling.final import cost_curve, policy_partition


def test_policy_holdout_is_later_and_labels_are_mature():
    data = pd.DataFrame({"cutoff": pd.date_range("2010-03-01", "2011-04-11", freq="W-MON")})
    fit, policy = policy_partition(data)
    assert policy.cutoff.nunique() == 6
    assert fit.cutoff.max() + pd.Timedelta(days=50) <= policy.cutoff.min()
    assert set(fit.cutoff).isdisjoint(policy.cutoff)


def test_curve_includes_never_contact_for_probability_one():
    data = pd.DataFrame({"RepurchaseWithin50Days": [0, 1], "p_class_1": [0, 1]})
    curve = cost_curve(data)
    assert curve.iloc[0].contacts == 2
    assert curve.iloc[-1].contacts == 0
    assert np.isinf(curve.iloc[-1].threshold)
    assert curve.mean_cost.min() == 5
