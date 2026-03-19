#!/usr/bin/env python3
"""Register S3 flight data in AWS Athena via the Glue Data Catalog.

Creates the flight_advisor database and the following external tables:
  - flights_processed   (processed/flights_processed.parquet)
  - train               (processed/train.parquet)
  - test                (processed/test.parquet)
  - airport_profiles    (refined/airport_profiles.parquet)

Usage:
    python src/aws/athena_client.py
    python src/aws/athena_client.py --bucket my-bucket --format parquet
    python src/aws/athena_client.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
    TokenRetrievalError,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATABASE = "flight_advisor"

# Schema definitions for each table.
# Each entry: (table_name, s3_prefix, [(col_name, col_type), ...])
TABLE_DEFINITIONS = [
    (
        "flights_processed",
        "processed",
        [
            ("YEAR", "INT"),
            ("MONTH", "INT"),
            ("DAY", "INT"),
            ("DAY_OF_WEEK", "INT"),
            ("AIRLINE", "STRING"),
            ("AIRLINE_NAME", "STRING"),
            ("FLIGHT_NUMBER", "INT"),
            ("TAIL_NUMBER", "STRING"),
            ("ORIGIN_AIRPORT", "STRING"),
            ("DESTINATION_AIRPORT", "STRING"),
            ("SCHEDULED_DEPARTURE", "INT"),
            ("DEPARTURE_TIME", "DOUBLE"),
            ("DEPARTURE_DELAY", "DOUBLE"),
            ("TAXI_OUT", "DOUBLE"),
            ("WHEELS_OFF", "DOUBLE"),
            ("SCHEDULED_TIME", "DOUBLE"),
            ("ELAPSED_TIME", "DOUBLE"),
            ("AIR_TIME", "DOUBLE"),
            ("DISTANCE", "DOUBLE"),
            ("WHEELS_ON", "DOUBLE"),
            ("TAXI_IN", "DOUBLE"),
            ("SCHEDULED_ARRIVAL", "INT"),
            ("ARRIVAL_TIME", "DOUBLE"),
            ("ARRIVAL_DELAY", "DOUBLE"),
            ("DIVERTED", "INT"),
            ("CANCELLED", "INT"),
            ("CANCELLATION_REASON", "STRING"),
            ("AIR_SYSTEM_DELAY", "DOUBLE"),
            ("SECURITY_DELAY", "DOUBLE"),
            ("AIRLINE_DELAY", "DOUBLE"),
            ("LATE_AIRCRAFT_DELAY", "DOUBLE"),
            ("WEATHER_DELAY", "DOUBLE"),
            ("ORIGIN_AIRPORT_NAME", "STRING"),
            ("ORIGIN_CITY", "STRING"),
            ("ORIGIN_STATE", "STRING"),
            ("ORIGIN_LATITUDE", "DOUBLE"),
            ("ORIGIN_LONGITUDE", "DOUBLE"),
            ("DESTINATION_AIRPORT_NAME", "STRING"),
            ("DESTINATION_CITY", "STRING"),
            ("DESTINATION_STATE", "STRING"),
            ("DESTINATION_LATITUDE", "DOUBLE"),
            ("DESTINATION_LONGITUDE", "DOUBLE"),
            ("TIME_OF_DAY", "STRING"),
            ("SEASON", "STRING"),
            ("IS_HOLIDAY", "INT"),
            ("ROUTE", "STRING"),
            ("IS_DELAYED", "INT"),
            ("AIRLINE_DOW", "STRING"),
            ("ORIGIN_DELAY_RATE", "DOUBLE"),
            ("DEST_DELAY_RATE", "DOUBLE"),
            ("CARRIER_DELAY_RATE", "DOUBLE"),
            ("ROUTE_DELAY_RATE", "DOUBLE"),
            ("CARRIER_DELAY_RATE_DOW", "DOUBLE"),
        ],
    ),
    (
        "train",
        "processed",
        [
            ("YEAR", "INT"),
            ("MONTH", "INT"),
            ("DAY", "INT"),
            ("DAY_OF_WEEK", "INT"),
            ("AIRLINE", "STRING"),
            ("ORIGIN_AIRPORT", "STRING"),
            ("DESTINATION_AIRPORT", "STRING"),
            ("ARRIVAL_DELAY", "DOUBLE"),
            ("IS_DELAYED", "INT"),
            ("TIME_OF_DAY", "STRING"),
            ("SEASON", "STRING"),
            ("IS_HOLIDAY", "INT"),
            ("ROUTE", "STRING"),
            ("AIRLINE_DOW", "STRING"),
            ("ORIGIN_DELAY_RATE", "DOUBLE"),
            ("DEST_DELAY_RATE", "DOUBLE"),
            ("CARRIER_DELAY_RATE", "DOUBLE"),
            ("ROUTE_DELAY_RATE", "DOUBLE"),
            ("CARRIER_DELAY_RATE_DOW", "DOUBLE"),
        ],
    ),
    (
        "test",
        "processed",
        [
            ("YEAR", "INT"),
            ("MONTH", "INT"),
            ("DAY", "INT"),
            ("DAY_OF_WEEK", "INT"),
            ("AIRLINE", "STRING"),
            ("ORIGIN_AIRPORT", "STRING"),
            ("DESTINATION_AIRPORT", "STRING"),
            ("ARRIVAL_DELAY", "DOUBLE"),
            ("IS_DELAYED", "INT"),
            ("TIME_OF_DAY", "STRING"),
            ("SEASON", "STRING"),
            ("IS_HOLIDAY", "INT"),
            ("ROUTE", "STRING"),
            ("AIRLINE_DOW", "STRING"),
            ("ORIGIN_DELAY_RATE", "DOUBLE"),
            ("DEST_DELAY_RATE", "DOUBLE"),
            ("CARRIER_DELAY_RATE", "DOUBLE"),
            ("ROUTE_DELAY_RATE", "DOUBLE"),
            ("CARRIER_DELAY_RATE_DOW", "DOUBLE"),
        ],
    ),
    (
        "airport_profiles",
        "refined",
        [
            ("IATA_CODE", "STRING"),
            ("AIRPORT", "STRING"),
            ("CITY", "STRING"),
            ("STATE", "STRING"),
            ("COUNTRY", "STRING"),
            ("LATITUDE", "DOUBLE"),
            ("LONGITUDE", "DOUBLE"),
            ("origin_flights", "BIGINT"),
            ("origin_delay_rate", "DOUBLE"),
            ("origin_avg_arrival_delay", "DOUBLE"),
            ("origin_avg_weather_delay", "DOUBLE"),
            ("origin_avg_late_aircraft_delay", "DOUBLE"),
            ("origin_avg_nas_delay", "DOUBLE"),
            ("origin_avg_security_delay", "DOUBLE"),
            ("origin_avg_carrier_delay", "DOUBLE"),
            ("dest_flights", "BIGINT"),
            ("dest_delay_rate", "DOUBLE"),
            ("dest_avg_arrival_delay", "DOUBLE"),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def load_env_file(path: Path = Path(".env")) -> None:
    """Load key=value pairs from *path* into os.environ (no overwrite)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register S3 flight data as Athena external tables."
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("S3_BUCKET") or os.getenv("S3_Bucket"),
        help="S3 bucket name (default: env S3_BUCKET)",
    )
    parser.add_argument(
        "--athena-results-prefix",
        default=os.getenv("S3_ATHENA_RESULTS_PREFIX", "athena-results"),
        help="S3 prefix for Athena query results (default: athena-results)",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("ATHENA_DATABASE", DATABASE),
        help=f"Athena database name (default: {DATABASE})",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="parquet",
        help="File format of the data in S3 (default: parquet)",
    )
    parser.add_argument(
        "--table-layout",
        choices=["file", "folder"],
        default=os.getenv("S3_TABLE_LAYOUT", "file"),
        help="S3 layout for tables: file uses <prefix>/<table>.<ext>; "
             "folder uses <prefix>/<table>/ (default: file)",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS region (default: env AWS_REGION or 'us-east-1')",
    )
    parser.add_argument(
        "--profile",
        default=os.getenv("AWS_PROFILE"),
        help="AWS profile to use (default: env AWS_PROFILE)",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop and recreate tables if they already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print DDL statements without executing them",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# AWS session helpers
# ---------------------------------------------------------------------------

def build_session(profile: str | None, region: str) -> boto3.Session:
    """Create a boto3 Session with a clear error when the profile is missing."""
    try:
        return boto3.Session(profile_name=profile, region_name=region)
    except ProfileNotFound:
        available = boto3.Session().available_profiles
        hint = f"Available profiles: {available}" if available else "No profiles found in ~/.aws/credentials."
        print(f"AWS profile '{profile}' not found. {hint}", file=sys.stderr)
        raise


def check_credentials(session: boto3.Session) -> None:
    """Fail fast when no valid AWS credentials are available."""
    creds = session.get_credentials()
    if creds is None:
        raise NoCredentialsError()
    resolved = creds.get_frozen_credentials()
    if not resolved.access_key:
        raise NoCredentialsError()


# ---------------------------------------------------------------------------
# DDL builders
# ---------------------------------------------------------------------------

def build_create_database_ddl(database: str) -> str:
    return f"CREATE DATABASE IF NOT EXISTS {database};"


def build_drop_table_ddl(database: str, table: str) -> str:
    return f"DROP TABLE IF EXISTS {database}.{table};"


def build_create_table_ddl(
    database: str,
    table: str,
    columns: list[tuple[str, str]],
    s3_location: str,
    file_format: str,
) -> str:
    col_defs = ",\n    ".join(f"`{name}` {dtype}" for name, dtype in columns)

    if file_format == "parquet":
        storage_clause = "STORED AS PARQUET"
    else:
        storage_clause = (
            "ROW FORMAT DELIMITED\n"
            "FIELDS TERMINATED BY ','\n"
            "LINES TERMINATED BY '\\n'"
        )

    tbl_props = (
        "TBLPROPERTIES (\n"
        "  'skip.header.line.count'='1',\n"
        "  'classification'='csv'\n"
        ")"
        if file_format == "csv"
        else "TBLPROPERTIES ('classification'='parquet')"
    )

    return (
        f"CREATE EXTERNAL TABLE IF NOT EXISTS {database}.{table} (\n"
        f"    {col_defs}\n"
        f")\n"
        f"{storage_clause}\n"
        f"LOCATION '{s3_location}'\n"
        f"{tbl_props};"
    )


def build_table_location(
    bucket: str,
    prefix: str,
    table: str,
    file_format: str,
    layout: str,
) -> str:
    base = f"s3://{bucket}/{prefix.strip('/')}"
    if layout == "folder":
        return f"{base}/{table}/"
    # layout == "file"
    ext = "parquet" if file_format == "parquet" else "csv"
    return f"{base}/{table}.{ext}"


# ---------------------------------------------------------------------------
# Athena execution helpers
# ---------------------------------------------------------------------------

def run_query(
    athena_client,
    sql: str,
    results_location: str,
    database: str | None = None,
    poll_interval: float = 1.0,
    timeout: int = 120,
) -> str:
    """Submit *sql* to Athena and block until it completes. Returns QueryExecutionId."""
    kwargs: dict = {
        "QueryString": sql,
        "ResultConfiguration": {"OutputLocation": results_location},
    }
    if database:
        kwargs["QueryExecutionContext"] = {"Database": database}

    response = athena_client.start_query_execution(**kwargs)
    exec_id = response["QueryExecutionId"]

    elapsed = 0.0
    while elapsed < timeout:
        status = athena_client.get_query_execution(QueryExecutionId=exec_id)
        state = status["QueryExecution"]["Status"]["State"]

        if state == "SUCCEEDED":
            return exec_id
        if state in ("FAILED", "CANCELLED"):
            reason = status["QueryExecution"]["Status"].get("StateChangeReason", "unknown")
            raise RuntimeError(f"Query {exec_id} {state}: {reason}\nSQL:\n{sql}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(f"Query {exec_id} did not complete within {timeout}s.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    load_env_file()
    args = parse_args()

    # ── Validate config ────────────────────────────────────────────────────
    if not args.bucket:
        print("Missing S3 bucket. Provide --bucket or set S3_BUCKET.", file=sys.stderr)
        return 2

    results_location = f"s3://{args.bucket}/{args.athena_results_prefix.strip('/')}/"

    if args.dry_run:
        print(f"-- Database\n{build_create_database_ddl(args.database)}\n")
        for table, prefix, columns in TABLE_DEFINITIONS:
            s3_loc = build_table_location(
                args.bucket,
                prefix,
                table,
                args.format,
                args.table_layout,
            )
            if args.drop_existing:
                print(f"-- Drop\n{build_drop_table_ddl(args.database, table)}\n")
            ddl = build_create_table_ddl(args.database, table, columns, s3_loc, args.format)
            print(f"-- Table: {table}\n{ddl}\n")
        print(f"-- Athena results location: {results_location}")
        return 0

    # ── Build session & validate credentials ──────────────────────────────
    try:
        session = build_session(args.profile, args.region)
        check_credentials(session)
    except ProfileNotFound:
        return 2
    except (NoCredentialsError, PartialCredentialsError):
        print(
            "AWS credentials not found or incomplete.\n"
            "Make sure one of the following is configured:\n"
            "  • ~/.aws/credentials  (run: aws configure)\n"
            "  • Environment variables: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY\n"
            "  • IAM role attached to the instance/container",
            file=sys.stderr,
        )
        return 2
    except TokenRetrievalError as exc:
        print(
            f"SSO/token retrieval failed: {exc}\n"
            "Try running: aws sso login --profile <profile>",
            file=sys.stderr,
        )
        return 2

    athena = session.client("athena")

    try:
        # ── Create database ────────────────────────────────────────────────
        print(f"[1/2] Creating database '{args.database}'...")
        run_query(
            athena,
            build_create_database_ddl(args.database),
            results_location,
        )
        print(f"      OK — database '{args.database}' is ready.")

        # ── Create tables ──────────────────────────────────────────────────
        print(f"\n[2/2] Registering {len(TABLE_DEFINITIONS)} tables...")
        for i, (table, prefix, columns) in enumerate(TABLE_DEFINITIONS, 1):
            s3_loc = f"s3://{args.bucket}/{prefix}/{table}/"

            if args.drop_existing:
                print(f"  [{i}/{len(TABLE_DEFINITIONS)}] Dropping '{table}'...")
                run_query(
                    athena,
                    build_drop_table_ddl(args.database, table),
                    results_location,
                    database=args.database,
                )

            ddl = build_create_table_ddl(args.database, table, columns, s3_loc, args.format)
            print(f"  [{i}/{len(TABLE_DEFINITIONS)}] Creating '{args.database}.{table}' -> {s3_loc}")
            run_query(athena, ddl, results_location, database=args.database)
            print(f"       OK")

        print(f"\nAll tables registered in database '{args.database}'.")
        print(f"Query results will be saved to: {results_location}")
        print(
            "\nExample query:\n"
            f"  SELECT origin_airport, COUNT(*) AS flights, AVG(arrival_delay) AS avg_delay\n"
            f"  FROM {args.database}.flights_processed\n"
            f"  GROUP BY origin_airport\n"
            f"  ORDER BY avg_delay DESC\n"
            f"  LIMIT 10;"
        )
        return 0

    except (RuntimeError, TimeoutError) as exc:
        print(f"\nAthena error: {exc}", file=sys.stderr)
        return 1
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        match code:
            case "AccessDeniedException":
                print(
                    "Access denied. Ensure your IAM role has:\n"
                    "  athena:StartQueryExecution, athena:GetQueryExecution\n"
                    "  glue:CreateDatabase, glue:CreateTable\n"
                    f"  s3:PutObject on s3://{args.bucket}/{args.athena_results_prefix}/",
                    file=sys.stderr,
                )
            case _:
                print(f"AWS error [{code}]: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
