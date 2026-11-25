"""Evaluate an existing monitored batch after 50 days of complete follow-up."""

import argparse
from pathlib import Path

from online_retail.logging import configure_logging
from online_retail.monitoring.outcomes import evaluate_outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--purchases", type=Path, default=Path("data/interim/purchases.parquet"))
    parser.add_argument("--coverage-end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    logger = configure_logging(args.log_file)
    try:
        report = evaluate_outcomes(args.run, args.purchases, args.coverage_end, args.output)
        logger.info(
            "outcomes_evaluated",
            extra={
                "context": {
                    "run_id": report["run_id"],
                    "cutoff": report["cutoff"],
                    "snapshots": report["overall"]["snapshots"],
                }
            },
        )
    except Exception:
        logger.exception("outcome_evaluation_failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
