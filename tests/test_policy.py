import pytest

from online_retail.modeling.policy import campaign_cost


def test_cost_matrix_orientation_and_threshold_boundary():
    result = campaign_cost([0, 0, 1, 1], [0.1, 0.5, 0.2, 0.9], 0.5, [[0, 15], [20, 10]])
    assert result == {"total_cost": 45.0, "mean_cost": 11.25, "contacts": 2}


def test_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError):
        campaign_cost([1], [1.1], 0.5, [[0, 10], [20, 10]])
