"""Honor the train/serve arguments supplied by SageMaker."""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "serve"])
    args = parser.parse_args()
    if args.command == "train":
        from online_retail.sagemaker.training import train

        train()
    else:
        # exec preserves signal delivery to the Gunicorn master (container PID 1).
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-m",
                "gunicorn",
                "--bind",
                "0.0.0.0:8080",
                "--workers",
                "1",
                "--timeout",
                "60",
                "--graceful-timeout",
                "20",
                "--worker-tmp-dir",
                "/tmp",
                "--error-logfile",
                "-",
                "online_retail.sagemaker.inference:create_app()",
            ],
        )


if __name__ == "__main__":
    main()
