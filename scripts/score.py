"""Score all known customers at a Monday cutoff; no training or target labels."""

import argparse
from pathlib import Path

from online_retail.logging import configure_logging
from online_retail.serving.workflow import run_scoring


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--purchases", type=Path, default=Path("data/interim/purchases.parquet"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cutoff", required=True, help="Timezone-naive Monday midnight")
    parser.add_argument(
        "--coverage-end", required=True, help="Exclusive ingestion coverage boundary"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    logger = configure_logging(args.log_file)
    try:
        run_scoring(
            args.purchases, args.bundle, args.cutoff, args.coverage_end, args.output, logger
        )
    except Exception:
        return 1  # Structured failure and traceback have already been logged.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
