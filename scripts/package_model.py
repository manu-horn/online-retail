"""Package a trusted frozen model/policy for local inference."""

import argparse
from pathlib import Path

from online_retail.logging import configure_logging
from online_retail.serving.bundle import create_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/final_roc_policy"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    logger = configure_logging(args.log_file)
    logger.info("packaging_started")
    try:
        metadata = create_bundle(args.source, args.output)
    except Exception:
        logger.exception("packaging_failed")
        return 1
    logger.info(
        "packaging_succeeded",
        extra={
            "context": {
                "output": str(args.output),
                "model_version": metadata["model_version"],
                "policy_version": metadata["policy_version"],
            }
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
