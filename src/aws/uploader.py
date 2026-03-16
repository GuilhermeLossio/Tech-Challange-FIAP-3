#!/usr/bin/env python3
"""Upload flight data to S3 raw zone.

Defaults to data/raw/flights.csv and uploads to s3://$S3_BUCKET/$S3_PREFIX/flights.csv.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import (
    ClientError,
    EndpointResolutionError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
    TokenRetrievalError,
)


def load_env_file(path: Path = Path(".env")) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload flight data (CSV) to the S3 raw zone."
    )
    parser.add_argument(
        "--input",
        default="data/raw/flights.csv",
        help="Path to flights CSV (default: data/raw/flights.csv)",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("S3_BUCKET") or os.getenv("S3_Bucket"),
        help="S3 bucket name (default: env S3_BUCKET)",
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("S3_PREFIX", "raw"),
        help="S3 prefix/folder (default: env S3_PREFIX or 'raw')",
    )
    parser.add_argument(
        "--name",
        default="flights.csv",
        help="S3 object name (default: flights.csv)",
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
        "--dry-run",
        action="store_true",
        help="Show the target S3 URI without uploading",
    )
    return parser.parse_args()


def build_session(profile: str | None, region: str) -> boto3.Session:
    """Create a boto3 Session, with a clear error if the profile doesn't exist."""
    try:
        return boto3.Session(profile_name=profile, region_name=region)
    except ProfileNotFound:
        profiles = boto3.Session().available_profiles
        hint = f"Available profiles: {profiles}" if profiles else "No profiles found in ~/.aws/credentials."
        print(f"AWS profile '{profile}' not found. {hint}", file=sys.stderr)
        raise


def check_credentials(session: boto3.Session) -> None:
    """Fail fast if no valid credentials are available."""
    creds = session.get_credentials()
    if creds is None:
        raise NoCredentialsError()
    # Force resolution of lazy credentials (e.g. SSO, assume-role)
    resolved = creds.get_frozen_credentials()
    if not resolved.access_key:
        raise NoCredentialsError()


def upload(s3_client, input_path: Path, bucket: str, key: str) -> None:
    """Upload a file to S3 with structured error handling."""
    s3_client.upload_file(str(input_path), bucket, key)


def main() -> int:
    load_env_file()
    args = parse_args()

    # ── Validate inputs ────────────────────────────────────────────────────────
    if not args.bucket:
        print("Missing S3 bucket. Provide --bucket or set S3_BUCKET.", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    prefix = args.prefix.strip("/") if args.prefix else ""
    key = f"{prefix}/{args.name}" if prefix else args.name
    s3_uri = f"s3://{args.bucket}/{key}"

    if args.dry_run:
        print(s3_uri)
        return 0

    # ── Build session & validate credentials ───────────────────────────────────
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

    # ── Upload ─────────────────────────────────────────────────────────────────
    s3 = session.client("s3")
    try:
        upload(s3, input_path, args.bucket, key)
    except FileNotFoundError:
        print(f"Input file disappeared before upload: {input_path}", file=sys.stderr)
        return 2
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        match code:
            case "NoSuchBucket":
                print(f"Bucket does not exist: {args.bucket}", file=sys.stderr)
            case "AccessDenied":
                print(
                    f"Access denied to s3://{args.bucket}/{key}.\n"
                    "Check that your IAM user/role has s3:PutObject permission on this bucket.",
                    file=sys.stderr,
                )
            case "InvalidAccessKeyId":
                print(
                    "Invalid AWS Access Key ID. Verify your credentials.",
                    file=sys.stderr,
                )
            case "ExpiredToken" | "ExpiredTokenException":
                print(
                    "AWS credentials have expired.\n"
                    "Refresh them with: aws sso login  (SSO) or aws configure (static keys).",
                    file=sys.stderr,
                )
            case _:
                print(f"S3 error [{code}]: {exc}", file=sys.stderr)
        return 1
    except EndpointResolutionError:
        print(
            f"Could not resolve S3 endpoint for region '{args.region}'.\n"
            "Check your --region value or AWS_REGION.",
            file=sys.stderr,
        )
        return 2

    print(f"Uploaded {input_path} -> {s3_uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
