"""Run tracked development training from the repository root."""

import argparse
from pathlib import Path

from online_retail.modeling.tracking import TrackingConfig
from online_retail.modeling.workflows import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["baseline", "elastic_net", "trees"])
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument(
        "--objective", choices=["weekly_roc_auc", "weekly_log_loss"], default="weekly_roc_auc"
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory to preserve earlier experiment artifacts")
    if args.trials < 1:
        parser.error("--trials must be positive")
    root = Path(__file__).resolve().parents[1]
    run_training(
        args.model, args.data, args.output, TrackingConfig.local(root), args.trials, args.objective
    )


if __name__ == "__main__":
    main()
