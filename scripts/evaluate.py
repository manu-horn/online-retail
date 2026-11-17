"""Run tracked diagnostics of an already frozen model; no threshold search."""

import argparse
from pathlib import Path

from online_retail.modeling.tracking import TrackingConfig
from online_retail.modeling.workflows import run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/final_roc_policy"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory")
    root = Path(__file__).resolve().parents[1]
    run_evaluation(args.data, args.model, args.output, TrackingConfig.local(root))


if __name__ == "__main__":
    main()
