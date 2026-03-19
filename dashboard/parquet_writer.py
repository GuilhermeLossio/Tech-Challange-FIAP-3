"""parquet_writer.py
====================
Converts processed flight data to partitioned Parquet on S3 (or local disk).

Usage
-----
# Write from a local CSV to S3
python parquet_writer.py \
    --input  data/flights_processed.csv \
    --bucket my-bucket \
    --prefix refined/flights_processed

# Write from a local CSV to a local directory (for testing)
python parquet_writer.py \
    --input  data/flights_processed.csv \
    --output local_parquet/flights_processed

After writing, register the Glue/Athena table once:
    python parquet_writer.py --repair-table --bucket my-bucket --prefix refined/flights_processed

Output layout
-------------
s3://<bucket>/<prefix>/year=2023/month=1/part-0.parquet
s3://<bucket>/<prefix>/year=2023/month=2/part-0.parquet
...
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Schema – explicit dtypes prevent the mixed-type warning and make Parquet
# files smaller / faster to read.
# ---------------------------------------------------------------------------
DTYPE_MAP: dict = {
    "year":                     "Int16",
    "month":                    "Int8",
    "day":                      "Int8",
    "day_of_week":              "Int8",
    "airline":                  "string",
    "airline_name":             "string",
    "flight_number":            "Int32",
    "tail_number":              "string",
    "origin_airport":           "string",
    "destination_airport":      "string",
    "scheduled_departure":      "Int32",
    "departure_time":           "Float32",
    "departure_delay":          "Float32",
    "taxi_out":                 "Float32",
    "wheels_off":               "Float32",
    "scheduled_time":           "Float32",
    "elapsed_time":             "Float32",
    "air_time":                 "Float32",
    "distance":                 "Float32",
    "wheels_on":                "Float32",
    "taxi_in":                  "Float32",
    "scheduled_arrival":        "Int32",
    "arrival_time":             "Float32",
    "arrival_delay":            "Float32",
    "diverted":                 "Int8",
    "cancelled":                "Int8",
    "cancellation_reason":      "string",
    "air_system_delay":         "Float32",
    "security_delay":           "Float32",
    "airline_delay":            "Float32",
    "late_aircraft_delay":      "Float32",
    "weather_delay":            "Float32",
    "origin_airport_name":      "string",
    "origin_city":              "string",
    "origin_state":             "string",
    "origin_latitude":          "Float32",
    "origin_longitude":         "Float32",
    "destination_airport_name": "string",
    "destination_city":         "string",
    "destination_state":        "string",
    "destination_latitude":     "Float32",
    "destination_longitude":    "Float32",
    "time_of_day":              "string",
    "season":                   "string",
    "is_holiday":               "Int8",
    "route":                    "string",
    "is_delayed":               "Int8",
    "airline_dow":              "string",
    "origin_delay_rate":        "Float32",
    "dest_delay_rate":          "Float32",
    "carrier_delay_rate":       "Float32",
    "route_delay_rate":         "Float32",
    "carrier_delay_rate_dow":   "Float32",
}

# Columns used for Hive-style partitioning in S3
PARTITION_COLS = ["year", "month"]

# Compression: snappy is fast and well-supported by Athena
PARQUET_COMPRESSION = "snappy"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load CSV with explicit dtypes to avoid DtypeWarning."""
    # Build dtype dict with only columns present in the file
    raw = pd.read_csv(path, nrows=0)
    available = {c.lower(): c for c in raw.columns}
    dtype_arg = {
        available[col]: dtype
        for col, dtype in DTYPE_MAP.items()
        if col in available
    }
    df = pd.read_csv(path, dtype=dtype_arg, low_memory=False)
    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    return df


def _s3_upload(local_path: Path, bucket: str, key: str, s3_client) -> None:
    print(f"  uploading → s3://{bucket}/{key}")
    s3_client.upload_file(str(local_path), bucket, key)


def write_partitioned_parquet(
    df: pd.DataFrame,
    output_dir: Path,
    bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    s3_client=None,
) -> list[str]:
    """
    Write df as Hive-partitioned Parquet files.

    If bucket + s3_prefix are given, files are uploaded to S3 after writing
    locally.  Returns list of written paths (local or s3://).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for (year, month), group in df.groupby(["year", "month"]):
        # Drop partition columns from the data file itself — Athena infers
        # them from the folder path.
        data = group.drop(columns=PARTITION_COLS, errors="ignore").reset_index(drop=True)
        table = pa.Table.from_pandas(data, preserve_index=False)

        partition_path = output_dir / f"year={year}" / f"month={month}"
        partition_path.mkdir(parents=True, exist_ok=True)
        local_file = partition_path / "part-0.parquet"

        pq.write_table(table, local_file, compression=PARQUET_COMPRESSION)
        print(f"  wrote {len(data):>7,} rows → {local_file}")

        if bucket and s3_prefix and s3_client:
            key = f"{s3_prefix.rstrip('/')}/year={year}/month={month}/part-0.parquet"
            _s3_upload(local_file, bucket, key, s3_client)
            written.append(f"s3://{bucket}/{key}")
        else:
            written.append(str(local_file))

    return written


def repair_athena_table(
    database: str,
    table: str,
    bucket: str,
    prefix: str,
    results_prefix: str = "athena-results",
    region: str = "us-east-1",
    profile: Optional[str] = None,
) -> None:
    """Run MSCK REPAIR TABLE so Athena discovers new partitions."""
    import time
    session = boto3.Session(profile_name=profile, region_name=region)
    athena = session.client("athena")
    s3_output = f"s3://{bucket}/{results_prefix.rstrip('/')}/"
    print(f"Running MSCK REPAIR TABLE {database}.{table} …")
    resp = athena.start_query_execution(
        QueryString=f"MSCK REPAIR TABLE `{database}`.`{table}`",
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": s3_output},
    )
    qid = resp["QueryExecutionId"]
    for _ in range(60):
        time.sleep(2)
        status = athena.get_query_execution(QueryExecutionId=qid)
        state = status["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            print(f"  → {state}")
            if state != "SUCCEEDED":
                reason = status["QueryExecution"]["Status"].get("StateChangeReason", "")
                print(f"  reason: {reason}", file=sys.stderr)
            return
    print("  → timed out waiting for MSCK REPAIR", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Write partitioned Parquet to S3 or local disk")
    p.add_argument("--input",  required=False, help="Path to source CSV file")
    p.add_argument("--bucket", required=False, help="S3 bucket name")
    p.add_argument("--prefix", required=False, default="refined/flights_processed",
                   help="S3 key prefix (default: refined/flights_processed)")
    p.add_argument("--output", required=False,
                   help="Local output directory (skips S3 upload, useful for testing)")
    p.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    p.add_argument("--profile", default=os.getenv("AWS_PROFILE"))
    p.add_argument("--repair-table", action="store_true",
                   help="Run MSCK REPAIR TABLE after writing (requires --bucket)")
    p.add_argument("--database", default=os.getenv("ATHENA_DATABASE", "flight_advisor"))
    p.add_argument("--table",    default=os.getenv("ATHENA_TABLE",    "flights_processed"))
    p.add_argument("--results-prefix", default=os.getenv("S3_ATHENA_RESULTS_PREFIX", "athena-results"))
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.repair_table:
        if not args.bucket:
            print("--bucket is required for --repair-table", file=sys.stderr)
            sys.exit(1)
        repair_athena_table(
            database=args.database,
            table=args.table,
            bucket=args.bucket,
            prefix=args.prefix,
            results_prefix=args.results_prefix,
            region=args.region,
            profile=args.profile,
        )
        return

    if not args.input:
        print("--input is required", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.input} …")
    df = load_csv(args.input)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    s3_client = None
    bucket = None
    s3_prefix = None

    if args.output:
        output_dir = Path(args.output)
    elif args.bucket:
        bucket = args.bucket
        s3_prefix = args.prefix
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        s3_client = session.client("s3")
        output_dir = Path(tempfile.mkdtemp()) / "parquet_staging"
    else:
        print("Provide --output (local) or --bucket (S3)", file=sys.stderr)
        sys.exit(1)

    print(f"Writing partitioned Parquet → {args.output or f's3://{bucket}/{s3_prefix}'}")
    paths = write_partitioned_parquet(df, output_dir, bucket, s3_prefix, s3_client)
    print(f"\nDone. {len(paths)} partition(s) written.")

    if args.bucket and not args.output:
        repair_athena_table(
            database=args.database,
            table=args.table,
            bucket=args.bucket,
            prefix=args.prefix,
            results_prefix=args.results_prefix,
            region=args.region,
            profile=args.profile,
        )


if __name__ == "__main__":
    main()