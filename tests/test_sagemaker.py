import json

import numpy as np
import pandas as pd
import pytest
from training_fixture import make_batch_records, make_reproduction_inputs

from online_retail.sagemaker.batch import CONTENT_TYPE, MAX_BYTES, prepare_batch
from online_retail.sagemaker.inference import create_app
from online_retail.sagemaker.training import train
from online_retail.serving.bundle import checksum, create_bundle


@pytest.fixture
def job(tmp_path):
    data, reference = make_reproduction_inputs(tmp_path)
    root = tmp_path / "opt/ml"
    (root / "input/data").mkdir(parents=True)
    (root / "input/data/training").symlink_to(data, target_is_directory=True)
    (root / "input/data/reference").symlink_to(reference, target_is_directory=True)
    (root / "input/config").mkdir()
    (root / "input/config/hyperparameters.json").write_text("{}")
    (root / "input/config/resourceconfig.json").write_text('{"hosts": ["local"]}')
    (root / "model").mkdir()
    return root, data, reference


def test_train_produces_directly_servable_bundle_without_test_rows(job):
    root, data, reference = job
    before = {p.name: checksum(p) for p in reference.iterdir()}
    report = train(root)
    assert report["verification"]["max_probability_difference"] == 0
    assert set(p.name for p in (root / "model").iterdir()) == {
        "model.pkl",
        "frozen_policy.json",
        "manifest.json",
    }
    assert before == {p.name: checksum(p) for p in reference.iterdir()}
    client = create_app(root / "model").test_client()
    assert client.get("/ping").status_code == 200
    payload = make_batch_records(data, root / "model")
    response = client.post("/invocations", data=payload, content_type=CONTENT_TYPE)
    assert response.status_code == 200
    scores = [json.loads(line) for line in response.text.splitlines()]
    assert [r["Customer ID"] for r in scores] == ["0", "1"]
    assert all(r["cutoff"].startswith("2010-07-19") for r in scores)
    assert all(isinstance(r["contact"], bool) for r in scores)
    assert client.get("/execution-parameters").json["MaxConcurrentTransforms"] == 1
    with pytest.raises(FileExistsError):
        train(root)
    assert (root / "output/failure").exists()


@pytest.mark.parametrize(
    "invalid", ["target", "feature_version", "coverage", "duplicate", "string"]
)
def test_invalid_snapshot_requests_are_rejected(job, invalid):
    root, data, reference = job
    create_bundle(reference, root / "bundle")
    client = create_app(root / "bundle").test_client()
    records = [json.loads(line) for line in make_batch_records(data, root / "bundle").splitlines()]
    if invalid == "target":
        records[0]["RepurchaseWithin50Days"] = 1
    elif invalid == "feature_version":
        records[0]["feature_version"] = "changed"
    elif invalid == "coverage":
        records[0]["coverage_end"] = "2010-07-18"
    elif invalid == "duplicate":
        records[1] = records[0]
    else:
        records[0]["recency_days"] = "2"
    response = client.post(
        "/invocations", data="\n".join(json.dumps(r) for r in records), content_type=CONTENT_TYPE
    )
    assert response.status_code == 400


def test_http_errors_and_bundle_integrity(job):
    root, data, reference = job
    create_bundle(reference, root / "bundle")
    client = create_app(root / "bundle").test_client()
    assert client.post("/invocations", data="{}", content_type="text/csv").status_code == 415
    assert (
        client.post("/invocations", data="not JSON", content_type=CONTENT_TYPE).status_code == 400
    )
    assert (
        client.post(
            "/invocations", data=b"x" * (MAX_BYTES + 1), content_type=CONTENT_TYPE
        ).status_code
        == 413
    )
    (root / "bundle/model.pkl").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        create_app(root / "bundle")


def test_training_rejects_overrides_and_records_failure(job):
    root, _, _ = job
    (root / "input/config/hyperparameters.json").write_text('{"max_depth": "9"}')
    with pytest.raises(ValueError, match="overrides"):
        train(root)
    assert (root / "output/failure").exists()
    assert not list((root / "model").iterdir())


def test_prepare_batch_matches_local_scoring(tmp_path):
    from container_smoke import synthetic_inputs

    from online_retail.serving.bundle import load_bundle
    from online_retail.serving.inference import predict_features, scoring_features

    inputs = tmp_path / "inputs"
    synthetic_inputs(inputs)
    output = tmp_path / "prepared"
    prepare_batch(
        inputs / "purchases.parquet", inputs / "bundle", "2011-06-06", "2011-06-06", output
    )
    client = create_app(inputs / "bundle").test_client()
    response = client.post(
        "/invocations", data=(output / "features.jsonl").read_bytes(), content_type=CONTENT_TYPE
    )
    assert response.status_code == 200
    result = [json.loads(line) for line in response.text.splitlines()]
    model, policy, _ = load_bundle(inputs / "bundle")
    features = scoring_features(
        pd.read_parquet(inputs / "purchases.parquet"), "2011-06-06", "2011-06-06"
    )
    expected = predict_features(model, policy, features)
    np.testing.assert_allclose(
        [r["purchase_probability"] for r in result],
        expected.purchase_probability,
        atol=1e-7,
        rtol=0,
    )
    assert [r["contact"] for r in result] == expected.contact.tolist()
    assert [r["Customer ID"] for r in result] == ["A", "B"]
    with pytest.raises(FileExistsError):
        prepare_batch(
            inputs / "purchases.parquet", inputs / "bundle", "2011-06-06", "2011-06-06", output
        )
