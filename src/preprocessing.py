#!/usr/bin/env python3
"""Preprocess flight data from S3 and write processed artifacts back to S3.

Pipeline (from PROTOTYPE.md):
- Join flights with airports and airlines
- Data cleaning (remove cancelled, handle nulls, remove outliers ARRIVAL_DELAY > 500)
- Feature engineering (TIME_OF_DAY, SEASON, IS_HOLIDAY, ROUTE)
- Target creation (IS_DELAYED = ARRIVAL_DELAY > 15)
- Historical target encoding computed on train only (avoid leakage)
- Train/test split stratified by IS_DELAYED
- Airport profiles aggregation for downstream analytics/RAG
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
    TokenRetrievalError,
)
from pandas.tseries.holiday import USFederalHolidayCalendar


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
        description="Preprocess flight data from S3 and write processed outputs to S3."
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("S3_BUCKET") or os.getenv("S3_Bucket"),
        help="S3 bucket name (default: env S3_BUCKET)",
    )
    parser.add_argument(
        "--raw-prefix",
        default=os.getenv("S3_RAW_PREFIX", "raw"),
        help="S3 raw prefix/folder (default: env S3_RAW_PREFIX or 'raw')",
    )
    parser.add_argument(
        "--processed-prefix",
        default=os.getenv("S3_PROCESSED_PREFIX", "processed"),
        help="S3 processed prefix (default: env S3_PROCESSED_PREFIX or 'processed')",
    )
    parser.add_argument(
        "--refined-prefix",
        default=os.getenv("S3_REFINED_PREFIX", "refined"),
        help="S3 refined prefix (default: env S3_REFINED_PREFIX or 'refined')",
    )
    parser.add_argument(
        "--dashboard-prefix",
        default=os.getenv("S3_DASHBOARD_PREFIX", "processed/flights_dashboard"),
        help="S3 prefix for the slim dashboard dataset (default: processed/flights_dashboard)",
    )
    parser.add_argument(
        "--athena-results-prefix",
        default=os.getenv("S3_ATHENA_RESULTS_PREFIX", "athena-results"),
        help="S3 prefix for Athena query results (default: athena-results)",
    )
    parser.add_argument(
        "--athena-database",
        default=os.getenv("ATHENA_DATABASE", "flight_advisor"),
        help="Athena database name (default: flight_advisor)",
    )
    parser.add_argument(
        "--athena-table",
        default=os.getenv("ATHENA_TABLE", "flights_processed"),
        help="Athena table name for partition sync (default: flights_processed)",
    )
    parser.add_argument(
        "--athena-table-layout",
        choices=["file", "folder"],
        default=os.getenv("S3_TABLE_LAYOUT", "file"),
        help="Athena table layout (default: file; partitioned tables forced to folder).",
    )
    parser.add_argument(
    "--flights-file",
    default="flights.csv",
    help="Flights filename in raw prefix (default: flights.csv)",
    )

    parser.add_argument(
        "--airports-file",
        default="airports.csv",
        help="Airports filename in raw prefix (default: airports.csv)",
    )

    parser.add_argument(
        "--airlines-file",
        default="airlines.csv",
        help="Airlines filename in raw prefix (default: airlines.csv)",
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
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N rows for quick runs (default: None = full data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned S3 reads/writes without executing",
    )
    parser.add_argument(
        "--skip-partitioned",
        action="store_true",
        help="Skip creating the partitioned flights_processed dataset on S3.",
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Skip creating the slim dashboard dataset on S3.",
    )
    parser.add_argument(
        "--skip-athena",
        action="store_true",
        help="Skip registering datasets in Athena.",
    )
    parser.add_argument(
        "--athena-drop-existing",
        action="store_true",
        help="Drop and recreate Athena tables if they already exist.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# AWS helpers
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


def s3_download(s3_client, bucket: str, key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading s3://{bucket}/{key} -> {dest}")
    try:
        s3_client.download_file(bucket, key, str(dest))
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in {"NoSuchKey", "404"}:
            print(
                f"Missing object in S3: s3://{bucket}/{key}\n"
                "Check that the file name and prefix are correct (e.g., raw/flights.parquet).",
                file=sys.stderr,
            )
        else:
            print(f"S3 download error [{code}]: {exc}", file=sys.stderr)
        raise


def s3_upload(s3_client, path: Path, bucket: str, key: str) -> None:
    print(f"  Uploading {path} -> s3://{bucket}/{key}")
    try:
        s3_client.upload_file(str(path), bucket, key)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        match code:
            case "NoSuchBucket":
                print(f"Bucket does not exist: {bucket}", file=sys.stderr)
            case "AccessDenied":
                print(
                    f"Access denied to s3://{bucket}/{key}.\n"
                    "Ensure your IAM user/role has s3:PutObject on this bucket.",
                    file=sys.stderr,
                )
            case "InvalidAccessKeyId":
                print("Invalid AWS Access Key ID. Verify your credentials.", file=sys.stderr)
            case "ExpiredToken" | "ExpiredTokenException":
                print(
                    "AWS credentials have expired.\n"
                    "Refresh with: aws sso login  (SSO) or aws configure  (static keys).",
                    file=sys.stderr,
                )
            case _:
                print(f"S3 error [{code}]: {exc}", file=sys.stderr)
        raise


def s3_delete_prefix(s3_client, bucket: str, prefix: str) -> None:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        items = page.get("Contents", [])
        if not items:
            continue
        objects = [{"Key": obj["Key"]} for obj in items]
        s3_client.delete_objects(Bucket=bucket, Delete={"Objects": objects})


def upload_parquet_folder(
    s3_client,
    local_path: Path,
    bucket: str,
    prefix: str,
) -> None:
    key_prefix = prefix.strip("/") + "/"
    try:
        s3_delete_prefix(s3_client, bucket, key_prefix)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        print(
            f"Warning: could not delete existing s3://{bucket}/{key_prefix} ({code}).",
            file=sys.stderr,
        )
    s3_upload(s3_client, local_path, bucket, f"{key_prefix}part-0.parquet")


def read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def coerce_df_to_athena_schema(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Coerce dataframe column types to match Athena table definitions."""
    try:
        from aws import athena_client as athena
    except ModuleNotFoundError:
        return df

    schema = next((cols for (tbl, _prefix, cols) in athena.TABLE_DEFINITIONS if tbl == table_name), None)
    if not schema:
        return df

    df = df.copy()
    for col, dtype in schema:
        if col not in df.columns:
            continue
        if dtype == "INT":
            df[col] = pd.to_numeric(df[col], errors="coerce").round(0).astype("Int32")
        elif dtype == "BIGINT":
            df[col] = pd.to_numeric(df[col], errors="coerce").round(0).astype("Int64")
        elif dtype == "DOUBLE":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        elif dtype == "STRING":
            df[col] = df[col].astype("string")
    return df


# ---------------------------------------------------------------------------
# Partitioned dataset helpers
# ---------------------------------------------------------------------------

DASHBOARD_COLS = {
    "YEAR": "year",
    "MONTH": "month",
    "DAY": "day",
    "ARRIVAL_DELAY": "arrival_delay",
    "IS_DELAYED": "is_delayed",
    "AIRLINE_NAME": "airline_name",
    "AIRLINE": "airline",
    "ORIGIN_STATE": "origin_state",
}


def build_dashboard_frame(flights: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in DASHBOARD_COLS if col not in flights.columns]
    if missing:
        raise ValueError(f"Missing columns for dashboard dataset: {', '.join(missing)}")
    return flights[list(DASHBOARD_COLS)].rename(columns=DASHBOARD_COLS).copy()


def write_partitioned_dataset_to_s3(
    df: pd.DataFrame,
    bucket: str,
    prefix: str,
    region: str,
    session: boto3.Session,
    partition_cols: list[str],
    lowercase: bool = False,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.fs as pafs
    except ModuleNotFoundError:
        print(
            "pyarrow is required to build partitioned datasets.\n"
            "Install with:\n"
            "  pip install pyarrow",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if lowercase:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        partition_cols = [c.lower() for c in partition_cols]

    table = pa.Table.from_pandas(df, preserve_index=False)
    creds = session.get_credentials()
    frozen = creds.get_frozen_credentials() if creds else None
    fs = pafs.S3FileSystem(
        access_key=frozen.access_key if frozen else None,
        secret_key=frozen.secret_key if frozen else None,
        session_token=frozen.token if frozen else None,
        region=region,
    )
    base_dir = f"{bucket}/{prefix.strip('/')}"
    file_format = ds.ParquetFileFormat()
    write_options = file_format.make_write_options(compression="snappy")
    ds.write_dataset(
        table,
        base_dir=base_dir,
        format=file_format,
        filesystem=fs,
        partitioning=partition_cols,
        partitioning_flavor="hive",
        file_options=write_options,
        existing_data_behavior="delete_matching",
    )


def register_athena_tables(
    session: boto3.Session,
    bucket: str,
    results_prefix: str,
    database: str,
    table_layout: str,
    drop_existing: bool,
    processed_prefix: str,
    refined_prefix: str,
    file_format: str = "parquet",
) -> None:
    try:
        from aws import athena_client as athena
    except ModuleNotFoundError:
        print(
            "Athena registration skipped: could not import aws.athena_client.",
            file=sys.stderr,
        )
        return

    athena_client = session.client("athena")
    results_location = f"s3://{bucket}/{results_prefix.strip('/')}/"

    athena.run_query(
        athena_client,
        athena.build_create_database_ddl(database),
        results_location,
    )

    effective_layout = table_layout
    if file_format == "parquet" and table_layout == "file":
        effective_layout = "folder"

    for table, prefix, columns in athena.TABLE_DEFINITIONS:
        if table in {"flights_processed", "train", "test"}:
            prefix = processed_prefix
        elif table == "airport_profiles":
            prefix = refined_prefix
        layout = athena.resolve_table_layout(table, effective_layout)
        s3_loc = athena.build_table_location(bucket, prefix, table, file_format, layout)

        if drop_existing:
            athena.run_query(
                athena_client,
                athena.build_drop_table_ddl(database, table),
                results_location,
                database=database,
            )

        partition_cols = athena.resolve_partition_cols(table, columns)
        ddl = athena.build_create_table_ddl(
            database,
            table,
            columns,
            s3_loc,
            file_format,
            partition_cols=partition_cols,
        )
        athena.run_query(athena_client, ddl, results_location, database=database)


def sync_athena_partitions(
    session: boto3.Session,
    bucket: str,
    results_prefix: str,
    database: str,
    table: str,
) -> None:
    try:
        from aws import athena_client as athena
    except ModuleNotFoundError:
        print(
            "Athena partition sync skipped: could not import aws.athena_client.",
            file=sys.stderr,
        )
        return

    athena_client = session.client("athena")
    results_location = f"s3://{bucket}/{results_prefix.strip('/')}/"
    catalog = os.getenv("ATHENA_CATALOG")
    if catalog:
        query = f"CALL {catalog}.system.sync_partition_metadata('{database}','{table}','ADD')"
    else:
        query = f"CALL system.sync_partition_metadata('{database}','{table}','ADD')"
    try:
        athena.run_query(athena_client, query, results_location, database=database)
        return
    except RuntimeError as exc:
        msg = str(exc)
        if "mismatched input 'CALL'" not in msg and "CALL" not in msg:
            raise

    # Fallback for Athena engine v2 (no CALL support)
    repair_sql = f"MSCK REPAIR TABLE {database}.{table};"
    athena.run_query(athena_client, repair_sql, results_location, database=database)


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def time_of_day(hour: int) -> str:
    if 5 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 16:
        return "afternoon"
    if 17 <= hour <= 21:
        return "evening"
    return "night"


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def build_holiday_set(dates: Iterable[pd.Timestamp]) -> set[pd.Timestamp]:
    dates = list(dates)
    if not dates:
        return set()
    start = min(dates).normalize()
    end = max(dates).normalize()
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start, end=end)
    return set(pd.to_datetime(holidays).normalize())


# ---------------------------------------------------------------------------
# Split & encoding
# ---------------------------------------------------------------------------

def stratified_split(
    df,
    target_col,
    test_size,
    seed
    ):
    df = df.sort_values(target_col).reset_index(drop=True)
    rng = np.random.default_rng(seed)

    train_idx, test_idx = [], []
    for _, group in df.groupby(target_col):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        split = int(len(indices) * (1 - test_size))
        train_idx.extend(indices[:split])
        test_idx.extend(indices[split:])

    return df.loc[train_idx], df.loc[test_idx]

def add_target_encodings(
    train_df: pd.DataFrame,
    df_to_apply: pd.DataFrame,
    target_col: str,
) -> Dict[str, pd.Series]:
    """Compute target-encoding rates on *train_df* and map onto *df_to_apply*.

    Computing rates exclusively on train_df prevents target leakage when
    df_to_apply is the test set.
    """
    global_rate = train_df[target_col].mean()

    def rate_map(group_col: str) -> pd.Series:
        rates = train_df.groupby(group_col)[target_col].mean()
        return df_to_apply[group_col].map(rates).fillna(global_rate)

    return {
        "ORIGIN_DELAY_RATE": rate_map("ORIGIN_AIRPORT"),
        "DEST_DELAY_RATE": rate_map("DESTINATION_AIRPORT"),
        "CARRIER_DELAY_RATE": rate_map("AIRLINE"),
        "ROUTE_DELAY_RATE": rate_map("ROUTE"),
        "CARRIER_DELAY_RATE_DOW": rate_map("AIRLINE_DOW"),
    }


# ---------------------------------------------------------------------------
# Airport profiles
# ---------------------------------------------------------------------------

# Delay columns sourced from the raw dataset.
# Adjust this list if your CSV uses different column names.
_DELAY_COLS = [
    "WEATHER_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "CARRIER_DELAY",
]


def build_airport_profiles(df: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-airport statistics for downstream analytics and RAG."""
    # Build origin aggregation dynamically — only include delay cols present in df
    extra_origin_aggs = {
        f"origin_avg_{col.lower()}": (col, "mean")
        for col in _DELAY_COLS
        if col in df.columns
    }
    origin_aggs = {
        "origin_flights": ("ROUTE", "count"),
        "origin_delay_rate": ("IS_DELAYED", "mean"),
        "origin_avg_arrival_delay": ("ARRIVAL_DELAY", "mean"),
        **extra_origin_aggs,
    }
    origin_stats = (
        df.groupby("ORIGIN_AIRPORT")
        .agg(**origin_aggs)
        .reset_index()
        .rename(columns={"ORIGIN_AIRPORT": "IATA_CODE"})
    )

    dest_stats = (
        df.groupby("DESTINATION_AIRPORT")
        .agg(
            dest_flights=("ROUTE", "count"),
            dest_delay_rate=("IS_DELAYED", "mean"),
            dest_avg_arrival_delay=("ARRIVAL_DELAY", "mean"),
        )
        .reset_index()
        .rename(columns={"DESTINATION_AIRPORT": "IATA_CODE"})
    )

    profiles = airports.merge(origin_stats, on="IATA_CODE", how="left").merge(
        dest_stats, on="IATA_CODE", how="left"
    )
    return profiles


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    load_env_file()
    args = parse_args()

    # ── Validate config ────────────────────────────────────────────────────
    if not args.bucket:
        print("Missing S3 bucket. Provide --bucket or set S3_BUCKET.", file=sys.stderr)
        return 2

    raw_prefix = args.raw_prefix.strip("/")
    processed_prefix = args.processed_prefix.strip("/")
    refined_prefix = args.refined_prefix.strip("/")
    dashboard_prefix = args.dashboard_prefix.strip("/")
    if args.dashboard_prefix == "processed/flights_dashboard" and processed_prefix != "processed":
        dashboard_prefix = f"{processed_prefix}/flights_dashboard"

    flights_key = f"{raw_prefix}/{args.flights_file}"
    airports_key = f"{raw_prefix}/{args.airports_file}"
    airlines_key = f"{raw_prefix}/{args.airlines_file}"

    out_processed_key = f"{processed_prefix}/flights_processed.parquet"
    out_train_key = f"{processed_prefix}/train.parquet"
    out_test_key = f"{processed_prefix}/test.parquet"
    out_profiles_key = f"{refined_prefix}/airport_profiles.parquet"
    out_refined_processed_key = f"{refined_prefix}/flights_processed.parquet"
    out_refined_train_key = f"{refined_prefix}/train.parquet"
    out_refined_test_key = f"{refined_prefix}/test.parquet"

    if args.dry_run:
        print("Planned S3 reads:")
        for k in (flights_key, airports_key, airlines_key):
            print(f"  s3://{args.bucket}/{k}")
        print("Planned S3 writes:")
        for k in (
            out_processed_key,
            out_train_key,
            out_test_key,
            out_profiles_key,
            out_refined_processed_key,
            out_refined_train_key,
            out_refined_test_key,
        ):
            print(f"  s3://{args.bucket}/{k}")
        if not args.skip_partitioned:
            print(f"  s3://{args.bucket}/{processed_prefix}/flights_processed/ (partitioned)")
        if not args.skip_dashboard:
            print(f"  s3://{args.bucket}/{dashboard_prefix}/ (dashboard)")
        if not args.skip_athena:
            print("Planned Athena folder layouts:")
            print(f"  s3://{args.bucket}/{processed_prefix}/train/ (parquet)")
            print(f"  s3://{args.bucket}/{processed_prefix}/test/ (parquet)")
            print(f"  s3://{args.bucket}/{refined_prefix}/airport_profiles/ (parquet)")
        if not args.skip_athena:
            print("Planned Athena registration:")
            print(f"  database: {args.athena_database}")
            print(f"  results:  s3://{args.bucket}/{args.athena_results_prefix.strip('/')}/")
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

    s3 = session.client("s3")

    # ── Download raw files ─────────────────────────────────────────────────
    tmp_in = Path(".tmp_preprocessing")
    tmp_out = Path(".tmp_preprocessing_out")
    try:
        print("\n[1/7] Downloading raw files from S3...")
        flights_path = tmp_in / args.flights_file
        airports_path = tmp_in / args.airports_file
        airlines_path = tmp_in / args.airlines_file

        s3_download(s3, args.bucket, flights_key, flights_path)
        s3_download(s3, args.bucket, airports_key, airports_path)
        s3_download(s3, args.bucket, airlines_key, airlines_path)

        # ── Load datasets ─────────────────────────────────────────────────
        flights = read_tabular(flights_path)
        airports = read_tabular(airports_path)
        airlines = read_tabular(airlines_path)
        print(f"  Loaded {len(flights):,} flight rows, {len(airports):,} airports, {len(airlines):,} airlines.")

        if args.sample:
            flights = flights.sample(n=min(args.sample, len(flights)), random_state=args.seed)
            print(f"  Sampled down to {len(flights):,} rows.")

        # ── Join ───────────────────────────────────────────────────────────
        print("\n[2/7] Joining datasets...")
        flights = flights.merge(
            airlines.rename(columns={"IATA_CODE": "AIRLINE", "AIRLINE": "AIRLINE_NAME"}),
            on="AIRLINE",
            how="left",
        )

        for side, col in (("ORIGIN", "ORIGIN_AIRPORT"), ("DESTINATION", "DESTINATION_AIRPORT")):
            small = airports.rename(columns={"IATA_CODE": col})
            flights = flights.merge(
                small[[col, "AIRPORT", "CITY", "STATE", "LATITUDE", "LONGITUDE"]].rename(
                    columns={
                        "AIRPORT":   f"{side}_AIRPORT_NAME",
                        "CITY":      f"{side}_CITY",
                        "STATE":     f"{side}_STATE",
                        "LATITUDE":  f"{side}_LATITUDE",
                        "LONGITUDE": f"{side}_LONGITUDE",
                    }
                ),
                on=col,
                how="left",
            )

        # ── Cleaning ───────────────────────────────────────────────────────
        print("\n[3/7] Cleaning data...")
        before = len(flights)
        flights["ARRIVAL_DELAY"] = pd.to_numeric(flights["ARRIVAL_DELAY"], errors="coerce")
        flights = flights[flights["CANCELLED"] != 1]
        flights = flights[flights["ARRIVAL_DELAY"].notna()]
        flights = flights[flights["ARRIVAL_DELAY"] <= 500]
        print(f"  Rows after cleaning: {len(flights):,}  (removed {before - len(flights):,})")

        # ── Feature engineering ────────────────────────────────────────────
        print("\n[4/7] Engineering features...")
        dep = flights["SCHEDULED_DEPARTURE"].fillna(0).astype(int)
        raw_hours = dep // 100
        invalid_count = (raw_hours > 23).sum()
        if invalid_count:
            print(f"  Warning: {invalid_count:,} rows had invalid departure hour — clipped to 23.")
        hours = raw_hours.clip(0, 23)
        flights["TIME_OF_DAY"] = hours.map(time_of_day)
        flights["SEASON"]      = flights["MONTH"].astype(int).map(season_from_month)

        flight_dates = pd.to_datetime(
            dict(year=flights["YEAR"], month=flights["MONTH"], day=flights["DAY"]),
            errors="coerce",
        )
        holiday_set = build_holiday_set(flight_dates.dropna())
        flights["IS_HOLIDAY"] = flight_dates.dt.normalize().isin(holiday_set).astype(int)

        # garantir tipos consistentes para parquet
        for col in ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "AIRLINE"]:
            flights[col] = flights[col].astype("string")

        flights["ROUTE"] = flights["ORIGIN_AIRPORT"] + "-" + flights["DESTINATION_AIRPORT"]
        flights["AIRLINE_DOW"] = flights["AIRLINE"] + "_" + flights["DAY_OF_WEEK"].astype(str)
        flights["IS_DELAYED"] = (flights["ARRIVAL_DELAY"] > 15).astype(int)

        delay_rate = flights["IS_DELAYED"].mean()
        print(f"  Overall delay rate: {delay_rate:.1%}")

        # ── Split & encode ─────────────────────────────────────────────────
        train_df, test_df = stratified_split(flights, "IS_DELAYED", args.test_size, args.seed)
        print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

        for df_part, enc in (
            (train_df, add_target_encodings(train_df, train_df, "IS_DELAYED")),
            (test_df,  add_target_encodings(train_df, test_df,  "IS_DELAYED")),
        ):
            for col, series in enc.items():
                df_part[col] = series

        for col, series in add_target_encodings(train_df, flights, "IS_DELAYED").items():
            flights[col] = series

        # ── Airport profiles ───────────────────────────────────────────────
        profiles = build_airport_profiles(flights, airports)

        # ── Coerce dtypes for Athena consistency ──────────────────────────
        flights = coerce_df_to_athena_schema(flights, "flights_processed")
        train_df = coerce_df_to_athena_schema(train_df, "train")
        test_df = coerce_df_to_athena_schema(test_df, "test")
        profiles = coerce_df_to_athena_schema(profiles, "airport_profiles")

        # ── Save locally ───────────────────────────────────────────────────
        tmp_out.mkdir(parents=True, exist_ok=True)
        processed_path = tmp_out / "flights_processed.parquet"
        train_path     = tmp_out / "train.parquet"
        test_path      = tmp_out / "test.parquet"
        profiles_path  = tmp_out / "airport_profiles.parquet"

        flights.to_parquet(processed_path, index=False)
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        profiles.to_parquet(profiles_path, index=False)

        # ── Upload ─────────────────────────────────────────────────────────
        print("\n[5/7] Uploading processed files to S3...")
        s3_upload(s3, processed_path, args.bucket, out_processed_key)
        s3_upload(s3, train_path,     args.bucket, out_train_key)
        s3_upload(s3, test_path,      args.bucket, out_test_key)
        s3_upload(s3, profiles_path,  args.bucket, out_profiles_key)

        # Mirror processed datasets into refined for downstream consumers
        s3_upload(s3, processed_path, args.bucket, out_refined_processed_key)
        s3_upload(s3, train_path,     args.bucket, out_refined_train_key)
        s3_upload(s3, test_path,      args.bucket, out_refined_test_key)

        # ── Partitioned datasets for Athena/Dashboard ─────────────────────
        print("\n[6/7] Building partitioned datasets for Athena/Dashboard...")
        if not args.skip_partitioned:
            write_partitioned_dataset_to_s3(
                flights,
                bucket=args.bucket,
                prefix=f"{processed_prefix}/flights_processed",
                region=args.region,
                session=session,
                partition_cols=["YEAR", "MONTH"],
                lowercase=False,
            )
        if not args.skip_dashboard:
            dash_df = build_dashboard_frame(flights)
            write_partitioned_dataset_to_s3(
                dash_df,
                bucket=args.bucket,
                prefix=dashboard_prefix,
                region=args.region,
                session=session,
                partition_cols=["year", "month"],
                lowercase=False,
            )
        if args.skip_partitioned and not args.skip_athena:
            print(
                "Warning: --skip-partitioned is set. Athena table 'flights_processed' expects "
                f"s3://{args.bucket}/{processed_prefix}/flights_processed/ partitions.",
                file=sys.stderr,
            )

        if not args.skip_athena:
            print("  Preparing folder layouts for Athena tables (train/test/airport_profiles)...")
            upload_parquet_folder(
                s3,
                train_path,
                args.bucket,
                f"{processed_prefix}/train",
            )
            upload_parquet_folder(
                s3,
                test_path,
                args.bucket,
                f"{processed_prefix}/test",
            )
            upload_parquet_folder(
                s3,
                profiles_path,
                args.bucket,
                f"{refined_prefix}/airport_profiles",
            )

        print("\n[7/7] Registering datasets in Athena...")
        if args.skip_athena:
            print("  Skipped.")
        else:
            try:
                register_athena_tables(
                    session=session,
                    bucket=args.bucket,
                    results_prefix=args.athena_results_prefix,
                    database=args.athena_database,
                    table_layout=args.athena_table_layout,
                    drop_existing=args.athena_drop_existing,
                    processed_prefix=processed_prefix,
                    refined_prefix=refined_prefix,
                )
                sync_athena_partitions(
                    session=session,
                    bucket=args.bucket,
                    results_prefix=args.athena_results_prefix,
                    database=args.athena_database,
                    table=args.athena_table,
                )
            except (RuntimeError, TimeoutError) as exc:
                print(f"\nAthena error: {exc}", file=sys.stderr)
                return 1

        print(f"\nDone. Processed data -> s3://{args.bucket}/{processed_prefix}/")
        print(f"      Refined data   -> s3://{args.bucket}/{refined_prefix}/")
        return 0

    except ClientError:
        # Individual upload errors already printed inside s3_upload
        return 1

    finally:
        # Always clean up temporary directories
        shutil.rmtree(tmp_in,  ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
