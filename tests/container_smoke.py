"""Standalone Docker contract check using synthetic data, not the retail dataset."""

import argparse
import json
import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from online_retail.features.snapshots import customer_features
from online_retail.serving.bundle import checksum, create_bundle
from online_retail.serving.workflow import run_scoring


def synthetic_inputs(folder):
    folder.mkdir()
    records = [
        ("A", "2010-12-01"),
        ("A", "2011-03-01"),
        ("A", "2011-05-15"),
        ("A", "2011-05-30"),
        ("B", "2011-06-05"),
        ("future", "2011-06-06"),
    ]
    purchases = pd.DataFrame(
        [
            {
                "Customer ID": customer,
                "Invoice": str(i),
                "InvoiceDate": pd.Timestamp(date),
                "StockCode": "A",
                "Quantity": 1,
                "Price": 10.0,
            }
            for i, (customer, date) in enumerate(records)
        ]
    )
    purchases.to_parquet(folder / "purchases.parquet", index=False)
    features = customer_features(purchases, "2011-06-06")
    columns = [c for c in features if c not in ("Customer ID", "cutoff")]
    model = Pipeline(
        [
            ("impute", SimpleImputer(add_indicator=True, keep_empty_features=True)),
            (
                "model",
                XGBClassifier(
                    n_estimators=10, max_depth=2, learning_rate=0.2, n_jobs=1, random_state=42
                ),
            ),
        ]
    )
    x = pd.concat([features[columns]] * 50, ignore_index=True)
    model.fit(x, [1, 0] * 50)
    source = folder / "source"
    source.mkdir()
    (source / "model.pkl").write_bytes(pickle.dumps(model))
    (source / "frozen_policy.json").write_text(
        json.dumps(
            {
                "feature_columns": columns,
                "threshold": 0.62,
                "fit_last_cutoff": "2011-01-10",
                "policy_last_cutoff": "2011-04-11",
                "purpose": "Synthetic transport fixture only; not an evaluated retail model",
            }
        )
    )
    create_bundle(source, folder / "bundle")


def check_container(image, report_path=None):
    with tempfile.TemporaryDirectory(prefix="retail-container-") as temp:
        root = Path(temp)
        inputs, outputs = root / "inputs", root / "outputs"
        synthetic_inputs(inputs)
        outputs.mkdir()
        run_scoring(
            inputs / "purchases.parquet",
            inputs / "bundle",
            "2011-06-06",
            "2011-06-06",
            root / "host-scores",
        )
        expected = pd.read_parquet(root / "host-scores/scores.parquet")
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
            "/tmp:rw,nosuid,size=128m",
            "--mount",
            f"type=bind,source={inputs},target=/input,readonly",
            "--mount",
            f"type=bind,source={outputs},target=/output",
        ]
        arguments = [
            "--purchases",
            "/input/purchases.parquet",
            "--bundle",
            "/input/bundle",
            "--cutoff",
            "2011-06-06",
            "--coverage-end",
            "2011-06-06",
            "--output",
            "/output/scores",
        ]
        result = subprocess.run(docker + [image] + arguments, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Container scoring failed: {result.stderr}")
        actual = pd.read_parquet(outputs / "scores/scores.parquet")
        assert_frame_equal(
            expected.drop(columns="run_id"),
            actual.drop(columns="run_id"),
            check_exact=False,
            atol=1e-7,
            rtol=1e-7,
        )
        events = [json.loads(line) for line in result.stderr.splitlines() if line.startswith("{")]
        assert events[-1]["event"] == "scoring_succeeded"
        # Packaging is also callable in the runtime image.
        packaged = subprocess.run(
            docker
            + [
                "--entrypoint",
                "python",
                image,
                "scripts/package_model.py",
                "--source",
                "/input/source",
                "--output",
                "/output/bundle",
            ],
            capture_output=True,
            text=True,
        )
        if packaged.returncode:
            raise RuntimeError(f"Container packaging failed: {packaged.stderr}")
        assert checksum(outputs / "bundle/model.pkl") == checksum(inputs / "bundle/model.pkl")
        assert checksum(outputs / "bundle/frozen_policy.json") == checksum(
            inputs / "bundle/frozen_policy.json"
        )
        # A retry cannot overwrite an existing output.
        retry = subprocess.run(docker + [image] + arguments, capture_output=True, text=True)
        assert retry.returncode == 1 and "scoring_failed" in retry.stderr, (
            f"Overwrite check exited {retry.returncode}:\n{retry.stdout}\n{retry.stderr}"
        )
        # Invalid declared coverage also fails without publishing partial output.
        invalid = arguments.copy()
        invalid[invalid.index("--coverage-end") + 1] = "2011-06-05"
        invalid[invalid.index("--output") + 1] = "/output/invalid"
        failure = subprocess.run(docker + [image] + invalid, capture_output=True, text=True)
        assert failure.returncode == 1 and "scoring_failed" in failure.stderr, (
            f"Coverage check exited {failure.returncode}:\n{failure.stdout}\n{failure.stderr}"
        )
        assert not (outputs / "invalid").exists()
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
                "from pathlib import Path; import importlib.util; "
                "assert all(not Path(p).exists() for p in "
                "['/app/data', '/app/artifacts', '/app/.aws']); "
                "assert all(importlib.util.find_spec(p) is None for p in "
                "['mlflow', 'shap', 'optuna', 'lightgbm'])",
            ],
            capture_output=True,
            text=True,
        )
        assert probe.returncode == 0, probe.stderr
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
        ).strip()
        report = {
            "image": image,
            "image_id": image_id,
            "fixture": "synthetic",
            "default_container_user": user,
            "runtime_isolation_verified": True,
            "snapshots": len(actual),
            "probability_parity": True,
            "max_probability_difference": float(
                np.abs(actual.purchase_probability - expected.purchase_probability).max()
            ),
            "read_only_network_disabled": True,
            "packaging_verified": True,
            "overwrite_rejected": True,
            "invalid_coverage_rejected": True,
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
    check_container(args.image, args.report)
