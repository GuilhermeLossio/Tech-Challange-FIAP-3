#!/usr/bin/env python3
"""Weekly prediction job for upcoming flights.

Optionally validates last week's predictions against actual outcomes
fetched from Athena, computing accuracy metrics and saving a report to S3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# Ensure src/ is on the import path when running from src/jobs
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import joblib
except ModuleNotFoundError:
    print(
        "Missing dependency: joblib. Install with:\n"
        "  pip install joblib",
        file=sys.stderr,
    )
    raise SystemExit(2)

from model import (
    FEATURE_SPECS,
    TARGET_COL,
    build_s3_client,
    coerce_feature_types,
    default_s3_uri,
    download_s3_object,
    is_s3_uri,
    load_env_file,
    load_csv_any,
    load_model_any,
    parse_s3_uri,
    upload_s3_object,
)


RATE_COLS = [
    "ORIGIN_DELAY_RATE",
    "DEST_DELAY_RATE",
    "CARRIER_DELAY_RATE",
    "ROUTE_DELAY_RATE",
    "CARRIER_DELAY_RATE_DOW",
]

# Athena query poll settings
_ATHENA_POLL_INTERVAL = 2.0   # seconds between status checks
_ATHENA_TIMEOUT       = 120   # seconds before giving up


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weekly prediction job for upcoming flights."
    )
    parser.add_argument("--model",        default=None, help="Path or s3:// URI to model.")
    parser.add_argument("--meta",         default=None, help="Optional metadata JSON (path or s3://).")
    parser.add_argument("--input",        default=None, help="Upcoming flights dataset (path or s3://).")
    parser.add_argument(
        "--rates-source",
        default=None,
        help="Historical dataset to compute delay rates (path or s3://).",
    )
    parser.add_argument("--output",       default=None, help="Output dataset (path or s3://).")
    parser.add_argument("--threshold",    type=float, default=0.5, help="Classification threshold.")
    parser.add_argument(
        "--week-start",
        default=None,
        help="Filter predictions by week start (YYYY-MM-DD).",
    )
    parser.add_argument("--week-days",    type=int, default=7, help="Days in the weekly window.")
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
        "--refined-prefix",
        default=os.getenv("S3_REFINED_PREFIX", "refined"),
        help="S3 refined prefix (default: env S3_REFINED_PREFIX or 'refined').",
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

    # ── Validation flags ───────────────────────────────────────────────────
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Compare last week's predictions with actual outcomes from Athena.",
    )
    parser.add_argument(
        "--athena-database",
        default=os.getenv("ATHENA_DATABASE", "flight_advisor"),
        help="Athena database to query actuals from (default: flight_advisor).",
    )
    parser.add_argument(
        "--athena-table",
        default=os.getenv("ATHENA_TABLE", "flights_processed"),
        help="Athena table with actual outcomes (default: flights_processed).",
    )
    parser.add_argument(
        "--athena-results-prefix",
        default=os.getenv("S3_ATHENA_RESULTS_PREFIX", "athena-results"),
        help="S3 prefix for Athena query results (default: athena-results).",
    )
    parser.add_argument(
        "--validation-output",
        default=None,
        help="Where to save the validation report dataset (path or s3://). "
             "Defaults to s3://<bucket>/<predictions-prefix>/validation_<tag>.csv",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path defaults
# ---------------------------------------------------------------------------

def default_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.model is None:
        if not args.bucket:
            raise ValueError("Missing --model and S3_BUCKET.")
        args.model = default_s3_uri(args.bucket, args.model_prefix, "delay_model.pkl")

    if args.rates_source is None:
        if not args.bucket:
            raise ValueError("Missing --rates-source and S3_BUCKET.")
        args.rates_source = default_s3_uri(args.bucket, args.processed_prefix, "train.parquet")

    if args.input is None:
        if not args.bucket:
            raise ValueError("Missing --input and S3_BUCKET.")
        args.input = default_s3_uri(args.bucket, args.refined_prefix, "future_flights.parquet")

    if args.output is None:
        if not args.bucket:
            raise ValueError("Missing --output and S3_BUCKET.")
        tag = (args.week_start or date.today().isoformat()).replace("-", "")
        args.output = default_s3_uri(args.bucket, args.predictions_prefix, f"weekly_predictions_{tag}.parquet")

    if args.validate and args.validation_output is None:
        if not args.bucket:
            raise ValueError("Missing --validation-output and S3_BUCKET.")
        tag = (args.week_start or date.today().isoformat()).replace("-", "")
        args.validation_output = default_s3_uri(
            args.bucket, args.predictions_prefix, f"validation_{tag}.parquet"
        )

    return args


# ---------------------------------------------------------------------------
# Feature engineering helpers (unchanged)
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
    from pandas.tseries.holiday import USFederalHolidayCalendar

    dates = list(dates)
    if not dates:
        return set()
    start = min(dates).normalize()
    end   = max(dates).normalize()
    cal   = USFederalHolidayCalendar()
    return set(pd.to_datetime(cal.holidays(start=start, end=end)).normalize())


def ensure_base_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required base columns: {', '.join(missing)}")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df  = df.copy()
    dep = df["SCHEDULED_DEPARTURE"].fillna(0).astype(int)
    hours = (dep // 100).clip(0, 23)
    df["TIME_OF_DAY"] = hours.map(time_of_day)
    df["SEASON"]      = df["MONTH"].astype(int).map(season_from_month)

    flight_dates = pd.to_datetime(
        dict(year=df["YEAR"], month=df["MONTH"], day=df["DAY"]), errors="coerce"
    )
    holiday_set      = build_holiday_set(flight_dates.dropna())
    df["IS_HOLIDAY"] = flight_dates.dt.normalize().isin(holiday_set).astype(int)

    df["ROUTE"]       = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)
    df["AIRLINE_DOW"] = df["AIRLINE"].astype(str) + "_" + df["DAY_OF_WEEK"].astype(str)

    df["PERIODO_DIA"] = df["TIME_OF_DAY"]
    df["ESTACAO"]     = df["SEASON"]
    df["IS_FERIADO"]  = df["IS_HOLIDAY"]
    df["ROTA"]        = df["ROUTE"]
    return df


def build_rate_maps(hist_df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], float]:
    if TARGET_COL not in hist_df.columns:
        raise ValueError(f"Historical dataset must include {TARGET_COL}.")

    global_rate = float(hist_df[TARGET_COL].mean())

    if "ROUTE" not in hist_df.columns:
        hist_df = hist_df.copy()
        hist_df["ROUTE"] = (
            hist_df["ORIGIN_AIRPORT"].astype(str) + "_" + hist_df["DESTINATION_AIRPORT"].astype(str)
        )
    if "AIRLINE_DOW" not in hist_df.columns:
        hist_df = hist_df.copy()
        hist_df["AIRLINE_DOW"] = (
            hist_df["AIRLINE"].astype(str) + "_" + hist_df["DAY_OF_WEEK"].astype(str)
        )

    maps = {
        "ORIGIN_DELAY_RATE":    hist_df.groupby("ORIGIN_AIRPORT")[TARGET_COL].mean().to_dict(),
        "DEST_DELAY_RATE":      hist_df.groupby("DESTINATION_AIRPORT")[TARGET_COL].mean().to_dict(),
        "CARRIER_DELAY_RATE":   hist_df.groupby("AIRLINE")[TARGET_COL].mean().to_dict(),
        "ROUTE_DELAY_RATE":     hist_df.groupby("ROUTE")[TARGET_COL].mean().to_dict(),
        "CARRIER_DELAY_RATE_DOW": hist_df.groupby("AIRLINE_DOW")[TARGET_COL].mean().to_dict(),
    }
    return maps, global_rate


def apply_rate_maps(
    df: pd.DataFrame, maps: Dict[str, Dict[str, float]], global_rate: float
) -> pd.DataFrame:
    df = df.copy()
    df["ORIGIN_DELAY_RATE"]      = df["ORIGIN_AIRPORT"].map(maps["ORIGIN_DELAY_RATE"]).fillna(global_rate)
    df["DEST_DELAY_RATE"]        = df["DESTINATION_AIRPORT"].map(maps["DEST_DELAY_RATE"]).fillna(global_rate)
    df["CARRIER_DELAY_RATE"]     = df["AIRLINE"].map(maps["CARRIER_DELAY_RATE"]).fillna(global_rate)
    df["ROUTE_DELAY_RATE"]       = df["ROUTE"].map(maps["ROUTE_DELAY_RATE"]).fillna(global_rate)
    df["CARRIER_DELAY_RATE_DOW"] = df["AIRLINE_DOW"].map(maps["CARRIER_DELAY_RATE_DOW"]).fillna(global_rate)
    df["ROTA_DELAY_RATE"]        = df["ROUTE_DELAY_RATE"]
    return df


def filter_week(df: pd.DataFrame, week_start: str, week_days: int) -> pd.DataFrame:
    start = date.fromisoformat(week_start)
    end   = start + timedelta(days=week_days)

    if "FLIGHT_DATE" in df.columns:
        dates = pd.to_datetime(df["FLIGHT_DATE"], errors="coerce").dt.date
    else:
        ensure_base_columns(df, ["YEAR", "MONTH", "DAY"])
        dates = pd.to_datetime(
            dict(year=df["YEAR"], month=df["MONTH"], day=df["DAY"]), errors="coerce"
        ).dt.date

    return df.loc[(dates >= start) & (dates < end)].copy()


def ensure_features(
    df: pd.DataFrame, rates_df: pd.DataFrame, required: List[str]
) -> pd.DataFrame:
    if not any(col not in df.columns for col in required):
        return df

    base_cols = [
        "YEAR", "MONTH", "DAY", "DAY_OF_WEEK",
        "SCHEDULED_DEPARTURE", "DISTANCE",
        "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "AIRLINE",
    ]
    ensure_base_columns(df, base_cols)
    df = add_engineered_features(df)

    if any(col not in df.columns for col in RATE_COLS):
        ensure_base_columns(rates_df, ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "AIRLINE", "DAY_OF_WEEK", TARGET_COL])
        maps, global_rate = build_rate_maps(rates_df)
        df = apply_rate_maps(df, maps, global_rate)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Missing required feature columns after engineering: " + ", ".join(missing))
    return df


def normalize_s3_parquet_uri(uri: str) -> str:
    if not is_s3_uri(uri):
        return uri
    if uri.lower().endswith(".parquet"):
        return uri
    if uri.lower().endswith(".csv"):
        return uri[:-4] + ".parquet"
    return uri + ".parquet"


def write_output(df: pd.DataFrame, output: str, s3, tmp_dir: Path) -> str:
    if is_s3_uri(output):
        output = normalize_s3_parquet_uri(output)
        bucket, key = parse_s3_uri(output)
        dest = tmp_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        upload_s3_object(s3, dest, bucket, key)
        return output
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return str(path)


def load_metadata(args, model_path: Path, s3, tmp_dir: Path) -> dict | None:
    candidates: List[Path] = []

    if args.meta:
        candidates.append(load_model_any(args.meta, s3, tmp_dir))
    else:
        candidates.extend([
            model_path.with_suffix(".json"),
            model_path.with_name(f"{model_path.stem}_meta.json"),
        ])
        if is_s3_uri(args.model) and s3:
            bucket, key = parse_s3_uri(args.model)
            key_path = PurePosixPath(key)
            for candidate in [
                key_path.with_suffix(".json"),
                key_path.with_name(f"{key_path.stem}_meta.json"),
            ]:
                dest = tmp_dir / str(candidate)
                if download_s3_object(s3, bucket, str(candidate), dest, allow_missing=True):
                    candidates.insert(0, dest)
                    break

    for candidate in candidates:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return None


def spec_columns(kind: str) -> List[str]:
    cols: List[str] = []
    for spec in FEATURE_SPECS:
        if spec.kind == kind:
            cols.append(spec.canonical)
            cols.extend(list(spec.alternatives))
    return cols


# ---------------------------------------------------------------------------
# Athena validation
# ---------------------------------------------------------------------------

def _run_athena_query(
    athena_client,
    sql: str,
    results_location: str,
    database: str,
) -> str:
    """Submit *sql* to Athena and block until it completes. Returns QueryExecutionId."""
    response = athena_client.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": results_location},
    )
    exec_id = response["QueryExecutionId"]

    elapsed = 0.0
    while elapsed < _ATHENA_TIMEOUT:
        status = athena_client.get_query_execution(QueryExecutionId=exec_id)
        state  = status["QueryExecution"]["Status"]["State"]
        if state == "SUCCEEDED":
            return exec_id
        if state in ("FAILED", "CANCELLED"):
            reason = status["QueryExecution"]["Status"].get("StateChangeReason", "unknown")
            raise RuntimeError(f"Athena query {state}: {reason}\nSQL:\n{sql}")
        time.sleep(_ATHENA_POLL_INTERVAL)
        elapsed += _ATHENA_POLL_INTERVAL

    raise TimeoutError(f"Athena query {exec_id} did not complete within {_ATHENA_TIMEOUT}s.")


def _athena_results_to_df(athena_client, exec_id: str) -> pd.DataFrame:
    """Page through Athena results and return a DataFrame."""
    paginator = athena_client.get_paginator("get_query_results")
    rows, headers = [], None
    for page in paginator.paginate(QueryExecutionId=exec_id):
        for i, row in enumerate(page["ResultSet"]["Rows"]):
            values = [c.get("VarCharValue", "") for c in row["Data"]]
            if headers is None:
                headers = values
            else:
                rows.append(values)
    return pd.DataFrame(rows, columns=headers) if headers else pd.DataFrame()


def fetch_actuals_from_athena(
    session,
    database: str,
    table: str,
    results_location: str,
    week_start: str,
    week_days: int,
    region: str,
) -> pd.DataFrame:
    """Query Athena for actual IS_DELAYED outcomes for a given week."""
    start = date.fromisoformat(week_start)
    end   = start + timedelta(days=week_days)

    sql = f"""
    SELECT
        YEAR,
        MONTH,
        DAY,
        AIRLINE,
        FLIGHT_NUMBER,
        ORIGIN_AIRPORT,
        DESTINATION_AIRPORT,
        SCHEDULED_DEPARTURE,
        ARRIVAL_DELAY,
        {TARGET_COL}
    FROM {database}.{table}
    WHERE
        CAST(YEAR  AS INTEGER) = {start.year}
        AND CAST(MONTH AS INTEGER) BETWEEN {start.month} AND {end.month}
        AND CAST(DAY   AS INTEGER) >= {start.day}
        AND CAST(DAY   AS INTEGER) <  {end.day}
        AND {TARGET_COL} IS NOT NULL
    """

    print(f"  Querying Athena: {database}.{table} for {week_start} → {end.isoformat()}...")
    athena = session.client("athena", region_name=region)
    exec_id = _run_athena_query(athena, sql, results_location, database)
    df = _athena_results_to_df(athena, exec_id)

    # Cast numeric columns
    for col in ["YEAR", "MONTH", "DAY", "FLIGHT_NUMBER", "SCHEDULED_DEPARTURE", "ARRIVAL_DELAY", TARGET_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Fetched {len(df):,} actual flight records from Athena.")
    return df


def compute_metrics(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """Join predictions with actuals and compute classification metrics.

    Join keys: YEAR, MONTH, DAY, AIRLINE, FLIGHT_NUMBER, ORIGIN_AIRPORT, DESTINATION_AIRPORT.
    Falls back to YEAR + MONTH + DAY + ROUTE if FLIGHT_NUMBER is absent.
    """
    join_keys = ["YEAR", "MONTH", "DAY", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
    if "FLIGHT_NUMBER" in predictions_df.columns and "FLIGHT_NUMBER" in actuals_df.columns:
        join_keys.append("FLIGHT_NUMBER")

    # Ensure key columns have matching types
    for col in join_keys:
        if col in predictions_df.columns:
            predictions_df[col] = pd.to_numeric(predictions_df[col], errors="coerce").fillna(
                predictions_df[col]
            )
        if col in actuals_df.columns:
            actuals_df[col] = pd.to_numeric(actuals_df[col], errors="coerce").fillna(
                actuals_df[col]
            )

    merged = predictions_df.merge(
        actuals_df[join_keys + [TARGET_COL, "ARRIVAL_DELAY"]].rename(
            columns={
                TARGET_COL:     "actual_is_delayed",
                "ARRIVAL_DELAY": "actual_arrival_delay",
            }
        ),
        on=join_keys,
        how="inner",
    )

    if merged.empty:
        print("  Warning: No matching flights found between predictions and actuals.")
        return merged, {}

    y_true = merged["actual_is_delayed"].astype(int)
    y_pred = merged["delay_prediction"].astype(int)
    y_prob = merged["delay_probability"].astype(float)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy  = (tp + tn) / len(y_true) if len(y_true) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Brier score (lower = better probability calibration)
    brier = float(np.mean((y_prob - y_true) ** 2))

    metrics = {
        "week_start":        args_week_start_placeholder,  # filled in main()
        "total_matched":     len(merged),
        "accuracy":          round(accuracy, 4),
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "f1_score":          round(f1, 4),
        "brier_score":       round(brier, 4),
        "true_positives":    tp,
        "true_negatives":    tn,
        "false_positives":   fp,
        "false_negatives":   fn,
        "actual_delay_rate": round(float(y_true.mean()), 4),
        "pred_delay_rate":   round(float(y_pred.mean()), 4),
    }

    return merged, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Placeholder replaced in main() — avoids a global variable
args_week_start_placeholder = ""


def main() -> int:
    load_env_file()
    args = parse_args()
    try:
        args = default_paths(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    needs_s3 = any(
        is_s3_uri(value)
        for value in (args.model, args.meta, args.input, args.rates_source, args.output)
        if value
    )
    s3      = build_s3_client(args.region, args.profile) if needs_s3 else None
    session = None
    if args.validate:
        import boto3
        session = (
            boto3.Session(profile_name=args.profile, region_name=args.region)
            if args.profile
            else boto3.Session(region_name=args.region)
        )
        if s3 is None:
            s3 = session.client("s3")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # ── Load model ─────────────────────────────────────────────────────
        model_path = load_model_any(args.model, s3, tmp_dir)
        pipeline   = joblib.load(model_path)
        meta       = load_metadata(args, model_path, s3, tmp_dir)

        if meta and "features" in meta and "selected" in meta["features"]:
            required = list(meta["features"]["selected"])
        else:
            required = [spec.canonical for spec in FEATURE_SPECS]

        # ── Load input data ────────────────────────────────────────────────
        try:
            future_df = load_csv_any(args.input, s3, tmp_dir)
            rates_df  = load_csv_any(args.rates_source, s3, tmp_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2

        if args.week_start:
            future_df = filter_week(future_df, args.week_start, args.week_days)

        # ── Feature engineering ────────────────────────────────────────────
        try:
            feature_df = ensure_features(future_df, rates_df, required)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        if TARGET_COL in feature_df.columns:
            feature_df = feature_df.drop(columns=[TARGET_COL])

        # ── Coerce types & predict ─────────────────────────────────────────
        if meta and "features" in meta and "selected" in meta["features"]:
            selected      = meta["features"]["selected"]
            missing_cols  = [col for col in selected if col not in feature_df.columns]
            if missing_cols:
                print(f"Missing columns required by model: {', '.join(missing_cols)}", file=sys.stderr)
                return 2
            numeric_cols     = meta["features"].get("numeric", [])
            categorical_cols = meta["features"].get("categorical", [])
            X = coerce_feature_types(feature_df[selected], numeric_cols, categorical_cols)
        else:
            numeric_cols     = [col for col in spec_columns("numeric")     if col in feature_df.columns]
            categorical_cols = [col for col in spec_columns("categorical") if col in feature_df.columns]
            X = coerce_feature_types(feature_df, numeric_cols, categorical_cols)

        probas = pipeline.predict_proba(X)[:, 1]
        preds  = (probas >= args.threshold).astype(int)

        output_df = future_df.copy()
        output_df["delay_probability"] = probas
        output_df["delay_prediction"]  = preds
        output_df["generated_at"]      = datetime.now(timezone.utc).isoformat()

        # ── Save predictions ───────────────────────────────────────────────
        final_output = write_output(output_df, args.output, s3, tmp_dir)
        print(f"Saved weekly predictions ({len(output_df):,} rows) to: {final_output}")

        # ── Validation against Athena actuals ──────────────────────────────
        if args.validate:
            if not args.week_start:
                # Default to the previous week
                today      = date.today()
                week_start = (today - timedelta(days=today.weekday() + 7)).isoformat()
            else:
                week_start = args.week_start

            results_location = (
                f"s3://{args.bucket}/{args.athena_results_prefix.strip('/')}/"
            )

            print(f"\n[Validation] Fetching actuals from Athena for week: {week_start}")
            try:
                actuals_df = fetch_actuals_from_athena(
                    session=session,
                    database=args.athena_database,
                    table=args.athena_table,
                    results_location=results_location,
                    week_start=week_start,
                    week_days=args.week_days,
                    region=args.region,
                )
            except (RuntimeError, TimeoutError) as exc:
                print(f"Athena error: {exc}", file=sys.stderr)
                return 1

            if actuals_df.empty:
                print("  No actuals found for this week. Skipping validation.")
                return 0

            # Load predictions for the same week (from output_df or S3)
            pred_week_df = filter_week(output_df, week_start, args.week_days) \
                if args.week_start else output_df

            merged_df, metrics = compute_metrics(pred_week_df, actuals_df)
            metrics["week_start"] = week_start  # fill placeholder

            if not metrics:
                print("  Could not compute metrics — no matched flights.")
                return 0

            # ── Print summary ──────────────────────────────────────────────
            print("\n── Validation Report ─────────────────────────────────────")
            print(f"  Week start       : {metrics['week_start']}")
            print(f"  Matched flights  : {metrics['total_matched']:,}")
            print(f"  Accuracy         : {metrics['accuracy']:.2%}")
            print(f"  Precision        : {metrics['precision']:.2%}")
            print(f"  Recall           : {metrics['recall']:.2%}")
            print(f"  F1 Score         : {metrics['f1_score']:.4f}")
            print(f"  Brier Score      : {metrics['brier_score']:.4f}  (lower = better)")
            print(f"  Actual delay rate: {metrics['actual_delay_rate']:.2%}")
            print(f"  Predicted delay  : {metrics['pred_delay_rate']:.2%}")
            print(f"  TP={metrics['true_positives']}  TN={metrics['true_negatives']}  "
                  f"FP={metrics['false_positives']}  FN={metrics['false_negatives']}")
            print("──────────────────────────────────────────────────────────")

            # ── Save validation report ─────────────────────────────────────
            if not merged_df.empty and args.validation_output:
                final_validation = write_output(merged_df, args.validation_output, s3, tmp_dir)
                print(f"\nValidation report saved to: {final_validation}")

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
