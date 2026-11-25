"""Reproduce the selected final fit, preserving the reference model and policy."""

import argparse
from pathlib import Path

from online_retail.modeling.tracking import TrackingConfig
from online_retail.modeling.workflows import run_reproduction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--reference", type=Path, default=Path("artifacts/final_roc_policy"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = run_reproduction(args.data, args.reference, args.output, TrackingConfig.local(root))
    print(f"Reproduced {report['fit_rows']:,} fitting snapshots; {report['verification']}")


if __name__ == "__main__":
    main()
