import json

import pandas as pd
import pytest
from training_fixture import make_reproduction_inputs

from online_retail.modeling.reproduction import fitting_population, reproduce_fit
from online_retail.serving.bundle import checksum


@pytest.fixture
def reproduction_inputs(tmp_path):
    return make_reproduction_inputs(tmp_path)


def test_reproduction_preserves_reference_and_matches_policy_without_test(reproduction_inputs):
    data, reference = reproduction_inputs
    before = {p.name: checksum(p) for p in reference.iterdir()}
    output = data.parent / "reproduced"
    report = reproduce_fit(data, reference, output)
    assert report["verification"]["max_probability_difference"] == 0
    assert report["verification"]["decisions_match"]
    assert not report["test_evaluated"] and not report["hyperparameter_search"]
    assert report["fit_rows"] == 8
    assert before == {p.name: checksum(p) for p in reference.iterdir()}
    assert (output / "frozen_policy.json").read_bytes() == (
        reference / "frozen_policy.json"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        reproduce_fit(data, reference, output)


def test_reproduction_rejects_changed_population(reproduction_inputs):
    data, reference = reproduction_inputs
    refit = pd.read_parquet(data / "refit.parquet").iloc[1:]
    refit.to_parquet(data / "refit.parquet", index=False)
    with pytest.raises(ValueError, match="population"):
        reproduce_fit(data, reference, data.parent / "invalid")
    assert not (data.parent / "invalid").exists()


def test_failed_parity_publishes_no_model(reproduction_inputs):
    data, reference = reproduction_inputs
    path = reference / "policy_predictions.parquet"
    predictions = pd.read_parquet(path)
    predictions["p_class_1"] = 0.99
    predictions.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="parity failed"):
        reproduce_fit(data, reference, data.parent / "invalid")
    assert not (data.parent / "invalid").exists()


def test_reproduction_rejects_label_overlap(reproduction_inputs):
    data, reference = reproduction_inputs
    frozen = json.loads((reference / "frozen_policy.json").read_text())
    frozen["policy_first_cutoff"] = "2010-03-29"
    with pytest.raises(ValueError, match="mature"):
        fitting_population(pd.read_parquet(data / "refit.parquet"), frozen)


def test_reproduction_is_tracked_as_fitting_not_import(reproduction_inputs):
    from mlflow.tracking import MlflowClient

    from online_retail.modeling.tracking import TrackingConfig
    from online_retail.modeling.workflows import run_reproduction

    data, reference = reproduction_inputs
    root = data.parent
    config = TrackingConfig(root, f"sqlite:///{root / 'mlflow.db'}", "reproduction-test")
    output = root / "tracked-reproduction"
    run_reproduction(data, reference, output, config)
    run_id = json.loads((output / "mlflow_run.json").read_text())["run_id"]
    run = MlflowClient(tracking_uri=config.uri).get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["workflow.mode"] == "fixed_parameter_reproduction"
    assert run.data.tags["reproduction.test_data_loaded"] == "false"
    assert run.data.metrics["reported.verification.max_probability_difference"] == 0
    assert run.data.tags["model.uri"]
