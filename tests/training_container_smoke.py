"""Synthetic fixed-fit container contract; no retail data or registry access."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from pandas.testing import assert_frame_equal
from training_fixture import make_reproduction_inputs

from online_retail.modeling.reproduction import reproduce_fit
from online_retail.serving.bundle import checksum


def check_training_container(image, report_path=None):
    with tempfile.TemporaryDirectory(prefix="retail-training-") as temp:
        root = Path(temp).resolve()
        data, reference = make_reproduction_inputs(root)
        original_checksums = {p.name: checksum(p) for p in reference.iterdir()}
        expected = root / "host-result"
        reproduce_fit(data, reference, expected)
        outputs = root / "outputs"
        outputs.mkdir()
        # Identical absolute paths keep SQLite/artifact URIs accessible from the host.
        docker = [
            "docker",
            "run",
            "--rm",
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
            "--mount",
            f"type=bind,source={data},target=/input/data,readonly",
            "--mount",
            f"type=bind,source={reference},target=/input/reference,readonly",
            "--mount",
            f"type=bind,source={outputs},target={outputs}",
        ]
        arguments = [
            "--data",
            "/input/data",
            "--reference",
            "/input/reference",
            "--output",
            str(outputs / "fit"),
            "--tracking-dir",
            str(outputs / "tracking"),
        ]
        result = subprocess.run(docker + [image] + arguments, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Training container failed:\n{result.stdout}\n{result.stderr}")
        actual = outputs / "fit"
        assert_frame_equal(
            pd.read_parquet(expected / "policy_predictions.parquet"),
            pd.read_parquet(actual / "policy_predictions.parquet"),
            check_exact=True,
        )
        assert original_checksums == {p.name: checksum(p) for p in reference.iterdir()}
        assert checksum(actual / "frozen_policy.json") == checksum(reference / "frozen_policy.json")
        tracking_uri = f"sqlite:///{outputs / 'tracking/mlflow.db'}"
        client = MlflowClient(tracking_uri=tracking_uri)
        run_id = json.loads((actual / "mlflow_run.json").read_text())["run_id"]
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        assert run.data.tags["workflow.mode"] == "fixed_parameter_reproduction"
        assert run.data.tags["reproduction.test_data_loaded"] == "false"
        assert run.data.tags["git.commit"] != "unknown"
        assert run.data.tags["git.dirty"] == "unknown"
        mlflow.set_tracking_uri(tracking_uri)
        # Reload from the HOST, proving that artifact paths survive container exit.
        model = mlflow.pyfunc.load_model(run.data.tags["model.uri"])
        features = pd.read_parquet(data / "refit.parquet")
        policy = json.loads((reference / "frozen_policy.json").read_text())
        features = features.loc[features.cutoff >= pd.Timestamp(policy["policy_first_cutoff"])]
        np.testing.assert_allclose(
            model.predict(features[policy["feature_columns"]]),
            pd.read_parquet(actual / "policy_predictions.parquet")[["p_class_0", "p_class_1"]],
            atol=1e-7,
            rtol=0,
        )
        retry = subprocess.run(docker + [image] + arguments, capture_output=True, text=True)
        assert retry.returncode != 0 and "FileExistsError" in retry.stderr
        user = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Config.User}}"], text=True
        ).strip()
        assert user and user.split(":")[0] not in {"0", "root"}
        probe = subprocess.run(
            docker
            + [
                "--entrypoint",
                "python",
                image,
                "-c",
                "from pathlib import Path; "
                "assert not Path('/app/data').exists(); "
                "assert not Path('/app/artifacts').exists(); "
                "assert not Path('/app/.aws').exists()",
            ],
            capture_output=True,
            text=True,
        )
        assert probe.returncode == 0, probe.stderr
        report = {
            "image": image,
            "image_id": subprocess.check_output(
                ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
            ).strip(),
            "fixture": "synthetic",
            "fit_rows": 8,
            "policy_rows": 4,
            "exact_prediction_parity": True,
            "reference_unchanged": True,
            "mlflow_host_reload": True,
            "overwrite_rejected": True,
            "test_rows_loaded": False,
            "default_container_user": user,
            "read_only_network_disabled": True,
        }
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    check_training_container(args.image, args.report)
