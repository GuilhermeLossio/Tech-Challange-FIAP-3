#!/usr/bin/env python3
"""Build a slim, partitioned Parquet dataset for the dashboard."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs

DEFAULT_COLUMNS = [
    "year",
    "month",
    "day",
    "arrival_delay",
    "is_delayed",
    "airline_name",
    "airline",
    "origin_state",
]


def parse_columns(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_COLUMNS.copy()
    cols = [c.strip().lower() for c in value.split(",")]
    return [c for c in cols if c]


def resolve_input_dataset(source: str, region: str) -> Tuple[ds.Dataset, pafs.FileSystem | None]:
    if source.lower().startswith("s3://"):
        s3 = pafs.S3FileSystem(region=region)
        return ds.dataset(source.removeprefix("s3://"), filesystem=s3, format="parquet"), s3
    return ds.dataset(source, format="parquet"), None


def resolve_output(
    output: str | None,
    bucket: str | None,
    prefix: str,
    region: str,
) -> Tuple[str, pafs.FileSystem | None]:
    if output:
        return output, None
    if not bucket:
        raise ValueError("Provide --output or --bucket.")
    key_prefix = prefix.strip("/")
    return f"{bucket}/{key_prefix}", pafs.S3FileSystem(region=region)


def resolve_columns(
    dataset: ds.Dataset,
    desired_lower: Iterable[str],
) -> Tuple[list[str], list[str]]:
    available = dataset.schema.names
    lookup = {name.lower(): name for name in available}
    actual: list[str] = []
    lower: list[str] = []
    missing: list[str] = []
    for col in desired_lower:
        found = lookup.get(col)
        if found:
            actual.append(found)
            lower.append(col)
        else:
            missing.append(col)

    if "year" not in lower or "month" not in lower:
        raise ValueError("Required columns missing: year/month not found in input dataset.")

    if missing:
        print(
            f"[warn] Columns not found and will be skipped: {', '.join(missing)}",
            file=sys.stderr,
        )
    return actual, lower


def build_reader(scanner: ds.Scanner, target_names: list[str]) -> pa.RecordBatchReader:
    schema_fields = []
    for idx, name in enumerate(target_names):
        field = scanner.schema.field(idx)
        schema_fields.append(pa.field(name, field.type, field.nullable, field.metadata))
    target_schema = pa.schema(schema_fields)

    def batch_iter():
        for batch in scanner.to_batches():
            yield batch.rename_columns(target_names)

    return pa.RecordBatchReader.from_batches(target_schema, batch_iter())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a slim, partitioned Parquet dataset for the dashboard."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Source Parquet file or dataset (local path or s3://).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Local output directory (skips S3 upload).",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("S3_BUCKET") or os.getenv("S3_Bucket"),
        help="S3 bucket name (default: env S3_BUCKET).",
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("S3_DASHBOARD_PREFIX", "processed/flights_dashboard"),
        help="S3 prefix for output dataset.",
    )
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated columns to keep (defaults to dashboard-required columns).",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS region.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200_000,
        help="Record batch size for streaming.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete matching output partitions before writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved input/output and exit.",
    )

    args = parser.parse_args()

    dataset, _ = resolve_input_dataset(args.input, args.region)
    desired = parse_columns(args.columns)
    actual_cols, lower_cols = resolve_columns(dataset, desired)
    output_base, output_fs = resolve_output(args.output, args.bucket, args.prefix, args.region)

    if args.dry_run:
        print(f"Input:  {args.input}")
        print(f"Output: {output_base}")
        print(f"Columns: {', '.join(lower_cols)}")
        return 0

    scanner = dataset.scanner(columns=actual_cols, batch_size=args.batch_size)
    reader = build_reader(scanner, lower_cols)

    file_format = ds.ParquetFileFormat()
    write_options = file_format.make_write_options(compression="snappy")
    existing_behavior = "delete_matching" if args.overwrite else "overwrite_or_ignore"

    ds.write_dataset(
        reader,
        base_dir=output_base,
        format=file_format,
        filesystem=output_fs,
        partitioning=["year", "month"],
        partitioning_flavor="hive",
        file_options=write_options,
        existing_data_behavior=existing_behavior,
    )

    print("Done.")
    if output_fs:
        print(f"S3 dataset -> s3://{output_base}/")
    else:
        print(f"Local dataset -> {output_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
