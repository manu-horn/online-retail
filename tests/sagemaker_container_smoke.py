"""Emulate train -> model.tar.gz -> serve -> Batch Transform, without AWS."""

import argparse
import json
import os
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
from training_fixture import make_batch_records, make_reproduction_inputs

from online_retail.serving.bundle import checksum


def checked(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def invoke(container, payload=None, content_type="application/jsonlines", path="/invocations"):
    code = """
import json, sys, urllib.request, urllib.error
body = sys.stdin.buffer.read() if sys.argv[2] == '/invocations' else None
request = urllib.request.Request('http://127.0.0.1:8080'+sys.argv[2], data=body,
                                headers={'Content-Type':sys.argv[1]})
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        print(json.dumps({'status':response.status, 'body':response.read().decode()}))
except urllib.error.HTTPError as error:
    print(json.dumps({'status':error.code, 'body':error.read().decode()}))
"""
    return json.loads(
        checked(
            ["docker", "exec", "-i", container, "python", "-c", code, content_type, path],
            input=payload or "",
            timeout=15,
        )
    )


def check_contracts(training_image, inference_image, report_path=None):
    with tempfile.TemporaryDirectory(prefix="retail-sagemaker-") as temp:
        root = Path(temp)
        data, reference = make_reproduction_inputs(root)
        before = {p.name: checksum(p) for p in reference.iterdir()}
        job = root / "job"
        for folder in ("model", "output", "config"):
            (job / folder).mkdir(parents=True)
        (job / "config/hyperparameters.json").write_text("{}")
        (job / "config/resourceconfig.json").write_text('{"hosts":["local"]}')
        (job / "config/inputdataconfig.json").write_text(
            '{"training":{"TrainingInputMode":"File"},"reference":{"TrainingInputMode":"File"}}'
        )
        runtime = [
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--tmpfs",
            "/tmp:rw,nosuid,size=256m",
        ]
        training = (
            ["docker", "run", "--rm"]
            + runtime
            + [
                "--mount",
                f"type=bind,source={data},target=/opt/ml/input/data/training,readonly",
                "--mount",
                f"type=bind,source={reference},target=/opt/ml/input/data/reference,readonly",
                "--mount",
                f"type=bind,source={job / 'config'},target=/opt/ml/input/config,readonly",
                "--mount",
                f"type=bind,source={job / 'model'},target=/opt/ml/model",
                "--mount",
                f"type=bind,source={job / 'output'},target=/opt/ml/output",
                training_image,
                "train",
            ]
        )
        checked(
            ["docker", "run", "--rm"]
            + runtime
            + [
                "--entrypoint",
                "python",
                training_image,
                "-c",
                "import importlib.util; "
                "assert all(importlib.util.find_spec(p) is None "
                "for p in ['mlflow', 'optuna', 'shap', 'lightgbm'])",
            ],
            timeout=30,
        )
        checked(training, timeout=120)
        verification = json.loads((job / "output/data/reproduction/report.json").read_text())
        assert verification["verification"]["max_probability_difference"] == 0
        assert before == {p.name: checksum(p) for p in reference.iterdir()}
        artifact = root / "model.tar.gz"
        with tarfile.open(artifact, "w:gz") as archive:
            for path in (job / "model").iterdir():
                archive.add(path, arcname=path.name)
        unpacked = root / "unpacked"
        with tarfile.open(artifact) as archive:
            archive.extractall(unpacked, filter="data")
        payload = make_batch_records(data, unpacked)
        launch = (
            ["docker", "run", "-d"]
            + runtime
            + [
                "--mount",
                f"type=bind,source={unpacked},target=/opt/ml/model,readonly",
                inference_image,
                "serve",
            ]
        )
        container = checked(launch).strip()
        try:
            last_error = None
            for _ in range(30):
                try:
                    health = invoke(container, path="/ping")
                    if health == {"status": 200, "body": ""}:
                        break
                except RuntimeError as error:
                    last_error = error
                time.sleep(0.5)
            else:
                raise RuntimeError(
                    f"Server never became healthy: {last_error}\n"
                    + checked(["docker", "logs", container])
                )
            result = invoke(container, payload)
            assert result["status"] == 200, result
            scores = [json.loads(line) for line in result["body"].splitlines()]
            assert [row["Customer ID"] for row in scores] == ["0", "1"]
            single = invoke(container, payload.splitlines()[0] + "\n")
            assert single["status"] == 200
            assert (
                json.loads(single["body"])["purchase_probability"]
                == scores[0]["purchase_probability"]
            )
            assert invoke(container, payload, content_type="text/csv")["status"] == 415
            assert invoke(container, '{"RepurchaseWithin50Days":1}\n')["status"] == 400
            assert invoke(container, path="/execution-parameters")["status"] == 200
            # Compare actual served probabilities with the trained pipeline.
            import pickle

            import pandas as pd

            with (unpacked / "model.pkl").open("rb") as handle:
                model = pickle.load(handle)
            records = pd.DataFrame([json.loads(line) for line in payload.splitlines()])
            columns = json.loads((unpacked / "frozen_policy.json").read_text())["feature_columns"]
            np.testing.assert_allclose(
                [row["purchase_probability"] for row in scores],
                model.predict_proba(records[columns].astype(float))[:, 1],
                atol=1e-7,
                rtol=0,
            )
            checked(["docker", "stop", "--time", "25", container], timeout=30)
            exit_code = checked(
                ["docker", "inspect", container, "--format", "{{.State.ExitCode}}"]
            ).strip()
            assert exit_code == "0", checked(["docker", "logs", container])
        finally:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        # Repeat training must fail rather than replace the model.
        retry = subprocess.run(training, capture_output=True, text=True, timeout=120)
        assert retry.returncode != 0, retry.stdout + retry.stderr
        assert (job / "output/failure").exists()
        report = {
            "training_image": training_image,
            "inference_image": inference_image,
            "fixture": "synthetic",
            "fit_rows": verification["fit_rows"],
            "training_parity": True,
            "minimal_training_dependencies": True,
            "artifact_roundtrip": True,
            "http_prediction_parity": True,
            "single_and_multi_record": True,
            "invalid_requests_rejected": True,
            "read_only_model": True,
            "graceful_shutdown": True,
            "original_artifacts_unchanged": True,
            "test_rows_loaded": False,
            "aws_resources_created": False,
        }
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-image", required=True)
    parser.add_argument("--inference-image", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    check_contracts(args.training_image, args.inference_image, args.report)
