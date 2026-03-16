#!/usr/bin/env python3
"""Training loop and model export wrapper."""

from __future__ import annotations

import argparse
import os

from model import load_env_file, train_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export the flight delay model."
    )
    parser.add_argument(
        "--train",
        default=None,
        help="Path or s3:// URI to train CSV (default: S3 when S3_BUCKET is set).",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Path or s3:// URI to test CSV (default: S3 when S3_BUCKET is set).",
    )
    parser.add_argument("--model-dir", default="models", help="Directory to save model artifacts.")
    parser.add_argument("--model-name", default="delay_model.pkl", help="Model filename.")
    parser.add_argument("--meta-name", default="delay_model_meta.json", help="Metadata filename.")
    parser.add_argument("--model-type", choices=["logreg", "rf", "xgb"], default="rf")
    parser.add_argument(
        "--from-s3",
        action="store_true",
        help="Load train/test from S3 (uses --bucket and --processed-prefix).",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("S3_BUCKET") or os.getenv("S3_Bucket"),
        help="S3 bucket name (default: env S3_BUCKET).",
    )
    parser.add_argument(
        "--processed-prefix",
        default=os.getenv("S3_PROCESSED_PREFIX", "processed"),
        help="S3 processed prefix (default: env S3_PROCESSED_PREFIX or 'processed').",
    )
    parser.add_argument(
        "--model-prefix",
        default=os.getenv("S3_MODEL_PREFIX", "models"),
        help="S3 prefix for model artifacts (default: env S3_MODEL_PREFIX or 'models').",
    )
    parser.add_argument("--train-key", default="train.csv", help="S3 object for train CSV.")
    parser.add_argument("--test-key", default="test.csv", help="S3 object for test CSV.")
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS region.",
    )
    parser.add_argument(
        "--profile",
        default=os.getenv("AWS_PROFILE"),
        help="AWS profile to use.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split when auto-splitting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--allow-missing", action="store_true", help="Allow missing features.")
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for metrics.")
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Do not upload model artifacts to S3 after training.",
    )

    parser.add_argument("--n-estimators", type=int, default=300, help="RF/XGB estimators.")
    parser.add_argument("--max-depth", type=int, default=8, help="RF/XGB max depth.")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="XGB learning rate.")
    parser.add_argument("--max-iter", type=int, default=300, help="LogReg max iterations.")
    parser.add_argument("--C", type=float, default=1.0, help="LogReg C.")

    return parser.parse_args()


def main() -> int:
    load_env_file()
    args = parse_args()
    return train_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
