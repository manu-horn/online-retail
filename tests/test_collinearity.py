import math

import pandas as pd
import pytest

from online_retail.analysis.collinearity import (
    categorical_associations,
    encoded_vif,
    mixed_associations,
)


def test_vif_detects_exact_linear_dependency_and_reference_codes_categories():
    data = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "category": ["low", "high", "low", "high", "low", "high"],
            "constant": [1] * 6,
        }
    )
    table, excluded = encoded_vif(data, list(data))
    assert excluded == ["constant"]
    values = table.set_index("feature").vif
    assert math.isinf(values["a"]) and math.isinf(values["b"])
    assert sum(name.startswith("category_") for name in values.index) == 1


def test_categorical_and_mixed_associations_have_expected_limits():
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"] * 10,
            "copy": ["X", "X", "Y", "Y"] * 10,
            "independent": ["yes", "no", "yes", "no"] * 10,
            "value": [1.0, 1.0, 10.0, 10.0] * 10,
        }
    )
    association = categorical_associations(data, ["group", "copy", "independent"])
    assert association.loc["group", "copy"] == pytest.approx(1.0)
    assert association.loc["group", "independent"] == pytest.approx(0.0)
    eta = mixed_associations(data, ["group", "independent"], ["value"])
    assert eta.loc["group", "value"] == pytest.approx(1.0)
    assert eta.loc["independent", "value"] == pytest.approx(0.0)
