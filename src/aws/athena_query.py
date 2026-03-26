#!/usr/bin/env python3
"""Run Athena queries and return pandas DataFrames."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd

try:
    import boto3
    from botocore.exceptions import (
        ClientError,
        NoCredentialsError,
        PartialCredentialsError,
        ProfileNotFound,
        TokenRetrievalError,
    )
except ModuleNotFoundError:
    print(
        "Missing dependency: boto3. Install with:\n"
        "  pip install boto3 botocore",
        file=sys.stderr,
    )
    raise SystemExit(2)


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


def build_session(profile, region):
    available = []
    try:
        available = boto3.Session().available_profiles
    except Exception:
        pass

    if profile and profile in available:
        return boto3.Session(profile_name=profile, region_name=region)
    else:
        return boto3.Session(region_name=region)

def check_credentials(session: boto3.Session) -> None:
    creds = session.get_credentials()
    if creds is None:
        raise NoCredentialsError()
    resolved = creds.get_frozen_credentials()
    if not resolved.access_key:
        raise NoCredentialsError()


def run_query(
    query: str,
    database: str,
    bucket: str,
    output_prefix: str,
    region: str,
    profile: str | None = None,
    max_rows: int | None = None,
    poll_seconds: float = 1.0,
    max_wait_seconds: float = 120.0,
) -> pd.DataFrame:
    session = build_session(profile, region)
    try:
        check_credentials(session)
    except (NoCredentialsError, PartialCredentialsError):
        raise RuntimeError("AWS credentials not found or incomplete.")
    except TokenRetrievalError as exc:
        raise RuntimeError(f"SSO/token retrieval failed: {exc}")

    client = session.client("athena")
    prefix = output_prefix.strip("/")
    output_location = f"s3://{bucket}/{prefix}/" if prefix else f"s3://{bucket}/"

    try:
        response = client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": database},
            ResultConfiguration={"OutputLocation": output_location},
        )
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        raise RuntimeError(f"Athena start_query_execution failed [{code}]: {exc}")

    qid = response["QueryExecutionId"]
    start = time.time()
    while True:
        status = client.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        state = status["State"]
        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            if state != "SUCCEEDED":
                reason = status.get("StateChangeReason", "Unknown error")
                raise RuntimeError(f"Athena query failed: {reason}")
            break
        if time.time() - start > max_wait_seconds:
            raise RuntimeError("Athena query timed out.")
        time.sleep(poll_seconds)

    rows: List[List[str | None]] = []
    headers: List[str] = []
    token = None
    while True:
        result = client.get_query_results(QueryExecutionId=qid, NextToken=token) if token else client.get_query_results(QueryExecutionId=qid)
        records = result.get("ResultSet", {}).get("Rows", [])
        for idx, record in enumerate(records):
            values = [col.get("VarCharValue") for col in record.get("Data", [])]
            if not headers:
                headers = values
                continue
            rows.append(values)
            if max_rows and len(rows) >= max_rows:
                break
        token = result.get("NextToken")
        if not token or (max_rows and len(rows) >= max_rows):
            break

    df = pd.DataFrame(rows, columns=headers)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted
    return df
