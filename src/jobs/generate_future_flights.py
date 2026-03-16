#!/usr/bin/env python3
"""Generate synthetic future_flights.csv for weekly predictions."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

# Ensure src/ is on the import path when running from src/jobs
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import (  # noqa: E402
    TARGET_COL,
    build_s3_client,
    default_s3_uri,
    is_s3_uri,
    load_env_file,
    load_model_any,
    parse_s3_uri,
    upload_s3_object,
)


BASE_COLS = [
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "SCHEDULED_DEPARTURE",
    "DISTANCE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "AIRLINE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic future_flights.csv for weekly predictions."
    )
    parser.add_argument("--input", default=None, help="Source CSV (path or s3://).")
    parser.add_argument("--output", default=None, help="Output CSV (path or s3://).")
    parser.add_argument("--rows", type=int, default=50000, help="Number of rows to generate.")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Week start date (YYYY-MM-DD). Defaults to next Monday.",
    )
    parser.add_argument("--week-days", type=int, default=7, help="Days in the weekly window.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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


def default_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.input is None:
        if not args.bucket:
            raise ValueError("Missing --input and S3_BUCKET. Provide a source CSV or set S3_BUCKET.")
        args.input = default_s3_uri(args.bucket, args.refined_prefix, "flights_processed.csv")

    if args.output is None:
        if not args.bucket:
            raise ValueError("Missing --output and S3_BUCKET. Provide an output path or set S3_BUCKET.")
        args.output = default_s3_uri(args.bucket, args.refined_prefix, "future_flights.csv")

    return args


def next_monday() -> date:
    today = date.today()
    days_ahead = (0 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def ensure_base_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required base columns: {', '.join(missing)}")


def reservoir_sample(
    path: Path,
    usecols: List[str],
    n: int,
    seed: int,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("--rows must be greater than 0.")

    rng = np.random.default_rng(seed)
    reservoir: List[pd.Series] = []
    total = 0

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        for _, row in chunk.iterrows():
            total += 1
            if len(reservoir) < n:
                reservoir.append(row)
                continue
            j = rng.integers(0, total)
            if j < n:
                reservoir[j] = row

    if not reservoir:
        return pd.DataFrame(columns=usecols)

    return pd.DataFrame(reservoir).reset_index(drop=True)


def assign_future_dates(
    df: pd.DataFrame,
    start: date,
    week_days: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [start + timedelta(days=i) for i in range(week_days)]
    chosen = rng.choice(dates, size=len(df), replace=True)

    df = df.copy()
    df["YEAR"] = [d.year for d in chosen]
    df["MONTH"] = [d.month for d in chosen]
    df["DAY"] = [d.day for d in chosen]
    df["DAY_OF_WEEK"] = [d.weekday() + 1 for d in chosen]
    df["FLIGHT_DATE"] = [d.isoformat() for d in chosen]
    return df


def write_output(df: pd.DataFrame, output: str, s3, tmp_dir: Path) -> None:
    if is_s3_uri(output):
        bucket, key = parse_s3_uri(output)
        dest = tmp_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)
        upload_s3_object(s3, dest, bucket, key)
        return

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    load_env_file()
    args = parse_args()
    try:
        args = default_paths(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    needs_s3 = any(is_s3_uri(value) for value in (args.input, args.output) if value)
    s3 = build_s3_client(args.region, args.profile) if needs_s3 else None

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        source_path = load_model_any(args.input, s3, tmp_dir)
        if not source_path.exists():
            print(f"Source CSV not found: {source_path}", file=sys.stderr)
            return 2

        try:
            sample_df = reservoir_sample(source_path, BASE_COLS, args.rows, args.seed)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        if sample_df.empty:
            print("Source CSV has no rows to sample.", file=sys.stderr)
            return 2

        ensure_base_columns(sample_df, BASE_COLS)
        if TARGET_COL in sample_df.columns:
            sample_df = sample_df.drop(columns=[TARGET_COL])

        start = date.fromisoformat(args.start_date) if args.start_date else next_monday()
        future_df = assign_future_dates(sample_df, start, args.week_days, args.seed)

        write_output(future_df, args.output, s3, tmp_dir)
        print(f"Generated future flights CSV -> {args.output}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
