"""Freeze a versioned aggregate reference from the last development-policy cutoff."""

import argparse
import tomllib
from pathlib import Path

from online_retail.monitoring.core import build_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/refit.parquet"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/monitoring.toml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = build_reference(
        args.data, args.bundle, args.output, tomllib.loads(args.config.read_text())
    )
    print(f"Reference: {reference['rows']:,} customers at {reference['cutoff']}; no labels used")


if __name__ == "__main__":
    main()
