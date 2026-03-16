#!/usr/bin/env python3
"""Generate future flights and run weekly predictions in one step."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

# Ensure src/ is on the import path when running from src/jobs
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import default_s3_uri, load_env_file  # noqa: E402

import generate_future_flights  # noqa: E402
import weekly_predict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate future_flights.csv and run weekly predictions."
    )
    parser.add_argument("--source", default=None, help="Source CSV for sampling (path or s3://).")
    parser.add_argument("--future-output", default=None, help="Future flights CSV (path or s3://).")
    parser.add_argument("--rows", type=int, default=50000, help="Number of rows to generate.")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Week start date for generated flights (YYYY-MM-DD).",
    )
    parser.add_argument("--week-days", type=int, default=7, help="Days in the weekly window.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--model", default=None, help="Model path or s3:// URI.")
    parser.add_argument("--meta", default=None, help="Optional metadata JSON (path or s3://).")
    parser.add_argument(
        "--rates-source",
        default=None,
        help="Historical CSV for delay rates (path or s3://).",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold.")
    parser.add_argument(
        "--week-start",
        default=None,
        help="Filter predictions by week start (YYYY-MM-DD).",
    )
    parser.add_argument("--output", default=None, help="Predictions output (path or s3://).")

    parser.add_argument(
        "--bucket",
        default=os.getenv("S3_BUCKET") or os.getenv("S3_Bucket"),
        help="S3 bucket name (default: env S3_BUCKET).",
    )
    parser.add_argument(
        "--refined-prefix",
        default=os.getenv("S3_REFINED_PREFIX", "refined"),
        help="S3 refined prefix (default: env S3_REFINED_PREFIX or 'refined').",
    )
    parser.add_argument(
        "--processed-prefix",
        default=os.getenv("S3_PROCESSED_PREFIX", "processed"),
        help="S3 processed prefix (default: env S3_PROCESSED_PREFIX or 'processed').",
    )
    parser.add_argument(
        "--model-prefix",
        default=os.getenv("S3_MODEL_PREFIX", "models"),
        help="S3 model prefix (default: env S3_MODEL_PREFIX or 'models').",
    )
    parser.add_argument(
        "--predictions-prefix",
        default=os.getenv("S3_PREDICTIONS_PREFIX", "predictions"),
        help="S3 predictions prefix (default: env S3_PREDICTIONS_PREFIX or 'predictions').",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS region (default: env AWS_REGION or 'us-east-1').",
    )
    parser.add_argument(
        "--profile",
        default=os.getenv("AWS_PROFILE"),
        help="AWS profile to use (default: env AWS_PROFILE).",
    )
    return parser.parse_args()


def build_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.source is None and args.bucket:
        args.source = default_s3_uri(args.bucket, args.refined_prefix, "flights_processed.csv")

    if args.future_output is None and args.bucket:
        args.future_output = default_s3_uri(args.bucket, args.refined_prefix, "future_flights.csv")

    if args.model is None and args.bucket:
        args.model = default_s3_uri(args.bucket, args.model_prefix, "delay_model.pkl")

    if args.rates_source is None and args.bucket:
        args.rates_source = default_s3_uri(args.bucket, args.processed_prefix, "train.csv")

    if args.output is None and args.bucket:
        tag = (args.week_start or args.start_date or date.today().isoformat()).replace("-", "")
        args.output = default_s3_uri(args.bucket, args.predictions_prefix, f"weekly_predictions_{tag}.csv")

    return args


def run_module(module, argv: list[str]) -> int:
    old_argv = sys.argv
    try:
        sys.argv = argv
        return int(module.main())
    finally:
        sys.argv = old_argv


def main() -> int:
    load_env_file()
    args = parse_args()
    args = build_defaults(args)

    if not args.source:
        print("Missing --source or S3_BUCKET for generator.", file=sys.stderr)
        return 2
    if not args.future_output:
        print("Missing --future-output or S3_BUCKET for generator.", file=sys.stderr)
        return 2
    if not args.model:
        print("Missing --model or S3_BUCKET for predictions.", file=sys.stderr)
        return 2
    if not args.rates_source:
        print("Missing --rates-source or S3_BUCKET for predictions.", file=sys.stderr)
        return 2
    if not args.output:
        print("Missing --output or S3_BUCKET for predictions.", file=sys.stderr)
        return 2

    gen_argv = [
        "generate_future_flights.py",
        "--input",
        args.source,
        "--output",
        args.future_output,
        "--rows",
        str(args.rows),
        "--week-days",
        str(args.week_days),
        "--seed",
        str(args.seed),
    ]
    if args.start_date:
        gen_argv.extend(["--start-date", args.start_date])
    if args.bucket:
        gen_argv.extend(["--bucket", args.bucket])
    if args.refined_prefix:
        gen_argv.extend(["--refined-prefix", args.refined_prefix])
    if args.processed_prefix:
        gen_argv.extend(["--processed-prefix", args.processed_prefix])
    if args.region:
        gen_argv.extend(["--region", args.region])
    if args.profile:
        gen_argv.extend(["--profile", args.profile])

    print("Running future flights generator...")
    code = run_module(generate_future_flights, gen_argv)
    if code != 0:
        return code

    predict_week_start = args.week_start or args.start_date
    pred_argv = [
        "weekly_predict.py",
        "--model",
        args.model,
        "--input",
        args.future_output,
        "--rates-source",
        args.rates_source,
        "--output",
        args.output,
        "--threshold",
        str(args.threshold),
        "--week-days",
        str(args.week_days),
    ]
    if predict_week_start:
        pred_argv.extend(["--week-start", predict_week_start])
    if args.meta:
        pred_argv.extend(["--meta", args.meta])
    if args.bucket:
        pred_argv.extend(["--bucket", args.bucket])
    if args.processed_prefix:
        pred_argv.extend(["--processed-prefix", args.processed_prefix])
    if args.refined_prefix:
        pred_argv.extend(["--refined-prefix", args.refined_prefix])
    if args.model_prefix:
        pred_argv.extend(["--model-prefix", args.model_prefix])
    if args.predictions_prefix:
        pred_argv.extend(["--predictions-prefix", args.predictions_prefix])
    if args.region:
        pred_argv.extend(["--region", args.region])
    if args.profile:
        pred_argv.extend(["--profile", args.profile])

    print("Running weekly predictions...")
    return run_module(weekly_predict, pred_argv)


if __name__ == "__main__":
    raise SystemExit(main())
