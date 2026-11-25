"""Export past-only snapshot JSON Lines for SageMaker Batch Transform."""

import argparse
from pathlib import Path

from online_retail.sagemaker.batch import prepare_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--purchases", type=Path, default=Path("data/interim/purchases.parquet"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--coverage-end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_batch(args.purchases, args.bundle, args.cutoff, args.coverage_end, args.output))


if __name__ == "__main__":
    main()
