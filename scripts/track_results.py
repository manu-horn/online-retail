"""Import saved models/reports into MLflow without training or test evaluation."""

import argparse
from pathlib import Path

from online_retail.modeling.tracking import TrackingConfig
from online_retail.modeling.workflows import import_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--name", default="saved-results")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(import_results(args.artifacts, args.data, TrackingConfig.local(root), args.name))


if __name__ == "__main__":
    main()
