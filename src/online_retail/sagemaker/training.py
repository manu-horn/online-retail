"""Single-instance File-mode reproduction using SageMaker's reserved paths."""

import json
import os
import shutil
import signal
from pathlib import Path

from online_retail.logging import configure_logging
from online_retail.modeling.reproduction import reproduce_fit
from online_retail.serving.bundle import create_bundle


def _stop(signum, frame):
    raise InterruptedError("Training stopped by SIGTERM")


def train(root=Path("/opt/ml")):
    root = Path(root)
    logger = configure_logging()
    output, model_dir = root / "output", root / "model"
    output.mkdir(parents=True, exist_ok=True)
    logger.info("sagemaker_training_started")
    previous = signal.signal(signal.SIGTERM, _stop)
    try:
        config = root / "input/config"
        hyperparameters = config / "hyperparameters.json"
        if hyperparameters.exists() and json.loads(hyperparameters.read_text()):
            raise ValueError("Fixed reproduction does not accept hyperparameter overrides")
        resources = config / "resourceconfig.json"
        if resources.exists() and len(json.loads(resources.read_text())["hosts"]) != 1:
            raise ValueError("Only single-instance training is supported")
        channels = config / "inputdataconfig.json"
        if channels.exists():
            for channel in json.loads(channels.read_text()).values():
                if channel.get("TrainingInputMode", "File") != "File":
                    raise ValueError("Only File-mode input channels are supported")
        if model_dir.exists() and any(model_dir.iterdir()):
            raise FileExistsError("Model destination must be empty")
        data = root / "input/data/training"
        reference = root / "input/data/reference"
        result_dir = output / "data/reproduction"
        report = reproduce_fit(data, reference, result_dir)
        bundle = output / "data/bundle"
        manifest = create_bundle(result_dir, bundle)
        model_dir.mkdir(parents=True, exist_ok=True)
        for name in ("model.pkl", "frozen_policy.json", "manifest.json"):
            shutil.copyfile(bundle / name, model_dir / name)
        provenance = {
            "workflow": "sagemaker_fixed_fit",
            "code_revision": os.environ.get("ONLINE_RETAIL_CODE_REVISION", "unknown"),
            "fit_rows": report["fit_rows"],
            "verification": report["verification"],
            "manifest": manifest,
            "tracking": "Portable job outputs; no container-local MLflow database",
        }
        (output / "data/job.json").write_text(json.dumps(provenance, indent=2) + "\n")
        logger.info(
            "sagemaker_training_succeeded",
            extra={
                "context": {
                    "fit_rows": report["fit_rows"],
                    "model_version": manifest["model_version"],
                }
            },
        )
        return provenance
    except Exception:
        (output / "failure").write_text("Fixed-fit training failed; inspect the job logs.")
        logger.exception("sagemaker_training_failed")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
