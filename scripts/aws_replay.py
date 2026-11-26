"""Prepare SageMaker request files offline or verify downloaded transform predictions."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def prepare(args):
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,39})", args.run):
        raise ValueError("Run name must be 1–40 lowercase letters, digits or hyphens.")
    for image in (args.training_image, args.inference_image):
        if not re.fullmatch(
            r"\d{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/"
            r"[a-z0-9_./-]+@sha256:[a-f0-9]{64}",
            image,
        ):
            raise ValueError("Supply private ECR image URIs pinned by digest.")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", args.bucket):
        raise ValueError("Supply an S3 bucket name.")
    base = f"s3://{args.bucket}/retail-replay/{args.run}"
    training = f"{args.run}-train"
    model = f"{args.run}-model"
    content_type = "application/jsonlines"
    requests = {
        "training": {
            "TrainingJobName": training,
            "AlgorithmSpecification": {
                "TrainingImage": args.training_image,
                "TrainingInputMode": "File",
            },
            "RoleArn": args.role,
            "EnableNetworkIsolation": True,
            "InputDataConfig": [
                {
                    "ChannelName": channel,
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"{base}/input/{channel}/",
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                }
                for channel in ("training", "reference")
            ],
            "ResourceConfig": {
                "InstanceType": args.training_instance,
                "InstanceCount": 1,
                "VolumeSizeInGB": 10,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 600},
            "OutputDataConfig": {"S3OutputPath": f"{base}/training-output"},
        },
        "model": {
            "ModelName": model,
            "ExecutionRoleArn": args.role,
            "EnableNetworkIsolation": True,
            "PrimaryContainer": {
                "Image": args.inference_image,
                "ModelDataUrl": f"{base}/training-output/{training}/output/model.tar.gz",
            },
        },
        "transform": {
            "TransformJobName": f"{args.run}-batch",
            "ModelName": model,
            "BatchStrategy": "MultiRecord",
            "MaxConcurrentTransforms": 1,
            "MaxPayloadInMB": 6,
            "ModelClientConfig": {"InvocationsTimeoutInSeconds": 60, "InvocationsMaxRetries": 0},
            "TransformInput": {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"{base}/input/batch/features.jsonl",
                    }
                },
                "ContentType": content_type,
                "SplitType": "Line",
            },
            "TransformOutput": {
                "S3OutputPath": f"{base}/predictions",
                "Accept": content_type,
                "AssembleWith": "Line",
            },
            "TransformResources": {"InstanceType": args.transform_instance, "InstanceCount": 1},
        },
    }
    args.output.mkdir(parents=True)
    for name, request in requests.items():
        (args.output / f"{name}.json").write_text(json.dumps(request, indent=2) + "\n")
    print(base)


def compare(args):
    keys = ["Customer ID", "cutoff"]
    fields = keys + [
        "purchase_probability",
        "contact",
        "policy_version",
        "feature_version",
        "data_version",
    ]
    local = pd.read_parquet(args.local, columns=fields)
    remote = pd.read_json(args.remote, lines=True, dtype={"Customer ID": str}, convert_dates=False)[
        fields
    ]
    for frame in (local, remote):
        frame["cutoff"] = pd.to_datetime(frame["cutoff"])
        if frame.empty or frame.isna().any().any():
            raise ValueError("Predictions must be nonempty and complete.")
    paired = local.merge(remote, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("Cloud and local customer/cutoff keys differ.")
    for key in fields[3:]:
        if not paired[f"{key}_x"].eq(paired[f"{key}_y"]).all():
            raise ValueError(f"Cloud and local {key} values differ.")
    expected, actual = paired.purchase_probability_x, paired.purchase_probability_y
    if not actual.between(0, 1).all() or not np.allclose(expected, actual, rtol=0, atol=1e-7):
        raise ValueError("Cloud predictions failed probability parity.")
    print(
        json.dumps(
            {"rows": len(paired), "max_probability_difference": float(abs(expected - actual).max())}
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    plan = commands.add_parser("prepare", help="Write request JSON without connecting to AWS")
    for name in (
        "run",
        "bucket",
        "role",
        "training-image",
        "inference-image",
        "training-instance",
        "transform-instance",
    ):
        plan.add_argument(f"--{name}", required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(action=prepare)
    verify = commands.add_parser(
        "compare", help="Compare downloaded JSON Lines against local Parquet"
    )
    verify.add_argument("--local", type=Path, required=True)
    verify.add_argument("--remote", type=Path, required=True)
    verify.set_defaults(action=compare)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
