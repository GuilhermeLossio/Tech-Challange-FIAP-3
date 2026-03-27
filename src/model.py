#!/usr/bin/env python3
"""Train, serialize, and run inference for the flight delay model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import joblib
except ModuleNotFoundError as exc:
    missing = exc.name or "scikit-learn"
    print(
        f"Missing dependency: {missing}. Install with:\n"
        "  pip install scikit-learn joblib",
        file=sys.stderr,
    )
    raise SystemExit(2)


TARGET_COL = "IS_DELAYED"


@dataclass(frozen=True)
class FeatureSpec:
    canonical: str
    alternatives: Tuple[str, ...]
    kind: str  # "numeric" or "categorical"


FEATURE_SPECS: List[FeatureSpec] = [
    FeatureSpec("MONTH", tuple(), "numeric"),
    FeatureSpec("DAY_OF_WEEK", tuple(), "numeric"),
    FeatureSpec("SCHEDULED_DEPARTURE", tuple(), "numeric"),
    FeatureSpec("DISTANCE", tuple(), "numeric"),
    FeatureSpec("TIME_OF_DAY", ("PERIODO_DIA",), "categorical"),
    FeatureSpec("SEASON", ("ESTACAO",), "categorical"),
    FeatureSpec("IS_HOLIDAY", ("IS_FERIADO",), "numeric"),
    FeatureSpec("ORIGIN_DELAY_RATE", tuple(), "numeric"),
    FeatureSpec("DEST_DELAY_RATE", tuple(), "numeric"),
    FeatureSpec("CARRIER_DELAY_RATE", tuple(), "numeric"),
    FeatureSpec("ROUTE_DELAY_RATE", ("ROTA_DELAY_RATE",), "numeric"),
    FeatureSpec("CARRIER_DELAY_RATE_DOW", tuple(), "numeric"),
    FeatureSpec("AIRLINE", tuple(), "categorical"),
]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, serialize, and run inference for the flight delay model."
    )
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="Train and serialize a model.")
    train.add_argument(
        "--train",
        default=None,
        help="Path or s3:// URI to train dataset (default: S3 when S3_BUCKET is set).",
    )
    train.add_argument(
        "--test",
        default=None,
        help="Path or s3:// URI to test dataset (default: S3 when S3_BUCKET is set).",
    )
    train.add_argument("--model-dir", default="models", help="Directory to save model artifacts.")
    train.add_argument("--model-name", default="delay_model.pkl", help="Model filename.")
    train.add_argument("--meta-name", default="delay_model_meta.json", help="Metadata filename.")
    train.add_argument("--model-type", choices=["logreg", "rf", "xgb"], default="rf")
    train.add_argument(
        "--from-s3",
        action="store_true",
        help="Load train/test from S3 (uses --bucket and --processed-prefix).",
    )
    train.add_argument(
        "--bucket",
        default=os.getenv("S3_BUCKET") or os.getenv("S3_Bucket"),
        help="S3 bucket name (default: env S3_BUCKET).",
    )
    train.add_argument(
        "--model-prefix",
        default=os.getenv("S3_MODEL_PREFIX", "models"),
        help="S3 prefix for model artifacts (default: env S3_MODEL_PREFIX or 'models').",
    )
    train.add_argument(
        "--processed-prefix",
        default=os.getenv("S3_PROCESSED_PREFIX", "processed"),
        help="S3 processed prefix (default: env S3_PROCESSED_PREFIX or 'processed').",
    )
    train.add_argument("--train-key", default="train.parquet", help="S3 object for train dataset.")
    train.add_argument("--test-key", default="test.parquet", help="S3 object for test dataset.")
    train.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS region (default: env AWS_REGION or 'us-east-1').",
    )
    train.add_argument(
        "--profile",
        default=os.getenv("AWS_PROFILE"),
        help="AWS profile to use (default: env AWS_PROFILE).",
    )
    train.add_argument("--test-size", type=float, default=0.2, help="Test split when auto-splitting.")
    train.add_argument("--seed", type=int, default=42, help="Random seed.")
    train.add_argument("--allow-missing", action="store_true", help="Allow missing features.")
    train.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    train.add_argument("--threshold", type=float, default=0.5, help="Threshold for metrics.")
    train.add_argument(
        "--no-upload",
        action="store_true",
        help="Do not upload model artifacts to S3 after training.",
    )

    train.add_argument("--n-estimators", type=int, default=300, help="RF/XGB estimators.")
    train.add_argument("--max-depth", type=int, default=8, help="RF/XGB max depth.")
    train.add_argument("--learning-rate", type=float, default=0.1, help="XGB learning rate.")
    train.add_argument("--max-iter", type=int, default=300, help="LogReg max iterations.")
    train.add_argument("--C", type=float, default=1.0, help="LogReg C.")

    pred = sub.add_parser("predict", help="Run inference with a serialized model.")
    pred.add_argument("--model", default="models/delay_model.pkl", help="Path to model .pkl.")
    pred.add_argument("--meta", default=None, help="Optional metadata JSON for validation.")
    pred.add_argument(
        "--input",
        default=None,
        help="CSV/Parquet input with feature columns (default: S3 when S3_BUCKET is set).",
    )
    pred.add_argument("--json", dest="json_path", default=None, help="JSON file input.")
    pred.add_argument("--row", default=None, help="Inline JSON row.")
    pred.add_argument("--output", default=None, help="Output CSV (default: stdout).")
    pred.add_argument("--threshold", type=float, default=0.5, help="Classification threshold.")
    pred.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS region (default: env AWS_REGION or 'us-east-1').",
    )
    pred.add_argument(
        "--profile",
        default=os.getenv("AWS_PROFILE"),
        help="AWS profile to use (default: env AWS_PROFILE).",
    )

    args = parser.parse_args()
    if args.command is None:
        args.command = "train"
    return args


def is_s3_uri(value: str) -> bool:
    return value.lower().startswith("s3://")


def default_s3_uri(bucket: str | None, prefix: str | None, filename: str) -> str | None:
    if not bucket:
        return None
    prefix = (prefix or "").strip("/")
    key = f"{prefix}/{filename}" if prefix else filename
    return f"s3://{bucket}/{key}"


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not is_s3_uri(uri):
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket_key = uri[5:]
    bucket, _, key = bucket_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"S3 URI must include bucket and key: {uri}")
    return bucket, key


def _available_boto3_profiles():
    import boto3

    saved = {key: os.environ.pop(key, None) for key in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE")}
    try:
        return boto3.Session().available_profiles
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def _session_without_env_profile(region: str):
    import boto3

    saved = {key: os.environ.pop(key, None) for key in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE")}
    try:
        return boto3.Session(region_name=region)
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def build_s3_client(region: str, profile: str | None):
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
            "boto3 is required for S3 access. Install with:\n"
            "  pip install boto3 botocore",
            file=sys.stderr,
        )
        raise SystemExit(2)

    requested_profile = (profile or "").strip()
    if requested_profile:
        try:
            session = boto3.Session(profile_name=requested_profile, region_name=region)
        except ProfileNotFound:
            available = _available_boto3_profiles()
            hint = (
                f"Available profiles: {available}"
                if available
                else "No profiles found in ~/.aws/credentials."
            )
            print(
                f"AWS profile '{requested_profile}' not found. "
                f"Falling back to environment or IAM role credentials. {hint}",
                file=sys.stderr,
            )
            session = _session_without_env_profile(region)
    else:
        session = _session_without_env_profile(region)

    try:
        creds = session.get_credentials()
        if creds is None:
            raise NoCredentialsError()
        resolved = creds.get_frozen_credentials()
        if not resolved.access_key:
            raise NoCredentialsError()
    except (NoCredentialsError, PartialCredentialsError):
        print(
            "AWS credentials not found or incomplete.\n"
            "Make sure one of the following is configured:\n"
            "  • ~/.aws/credentials  (run: aws configure)\n"
            "  • Environment variables: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY\n"
            "  • IAM role attached to the instance/container",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except TokenRetrievalError as exc:
        print(
            f"SSO/token retrieval failed: {exc}\n"
            "Try running: aws sso login --profile <profile>",
            file=sys.stderr,
        )
        raise SystemExit(2)

    return session.client("s3")


def download_s3_object(
    s3,
    bucket: str,
    key: str,
    dest: Path,
    allow_missing: bool = False,
) -> bool:
    from botocore.exceptions import ClientError

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(bucket, key, str(dest))
        return True
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in {"NoSuchKey", "404"}:
            if allow_missing:
                return False
            print(f"Missing object in S3: s3://{bucket}/{key}", file=sys.stderr)
        else:
            print(f"S3 download error [{code}]: {exc}", file=sys.stderr)
        raise SystemExit(2)


def upload_s3_object(s3, path: Path, bucket: str, key: str) -> None:
    from botocore.exceptions import ClientError

    try:
        s3.upload_file(str(path), bucket, key)
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
        raise SystemExit(1)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def stratified_split(
    df: pd.DataFrame, target_col: str, test_size: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(target_col).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for _, group in df.groupby(target_col):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        split = int(len(indices) * (1 - test_size))
        train_idx.extend(indices[:split])
        test_idx.extend(indices[split:])
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def resolve_s3_sources(args: argparse.Namespace) -> Tuple[str, str | None]:
    if args.from_s3:
        if not args.bucket:
            raise ValueError("Missing --bucket for --from-s3.")
        prefix = args.processed_prefix.strip("/") if args.processed_prefix else ""
        train_key = f"{prefix}/{args.train_key}" if prefix else args.train_key
        test_key = f"{prefix}/{args.test_key}" if prefix else args.test_key
        return f"s3://{args.bucket}/{train_key}", f"s3://{args.bucket}/{test_key}"
    train_source = args.train or default_s3_uri(
        args.bucket, args.processed_prefix, args.train_key
    )
    if args.test is not None:
        test_source = args.test
    elif args.train is None:
        test_source = default_s3_uri(args.bucket, args.processed_prefix, args.test_key)
    else:
        test_source = None
    if not train_source:
        raise ValueError("Missing --train and S3_BUCKET. Provide a file path or set S3_BUCKET.")
    return train_source, test_source


def load_csv_any(
    source: str,
    s3,
    tmp_dir: Path,
    allow_missing: bool = False,
) -> pd.DataFrame | None:
    """
    Loads data from a source (local or S3), with a fallback from .parquet to .csv.
    """

    def _try_load(current_source: str) -> pd.DataFrame | None:
        """Tries to load a single source, returns None on failure."""
        if is_s3_uri(current_source):
            if not s3:
                # This can happen if a local-only execution gets an s3 path by mistake
                print(f"ERROR: S3 URI '{current_source}' found but S3 is not configured.", file=sys.stderr)
                return None
            bucket, key = parse_s3_uri(current_source)
            dest = tmp_dir / PurePosixPath(key)
            if not download_s3_object(s3, bucket, key, dest, allow_missing=True):
                return None
            return load_csv(dest)

        path = Path(current_source)
        if not path.exists():
            return None
        return load_csv(path)

    # Try original source
    df = _try_load(source)
    if df is not None:
        return df

    # If not found, try fallback from .parquet to .csv
    if source.endswith(".parquet"):
        fallback_source = source.replace(".parquet", ".csv")
        print(f"INFO: '{source}' not found, falling back to '{fallback_source}'.", file=sys.stderr)
        df = _try_load(fallback_source)
        if df is not None:
            return df

    # If still not found, handle final error
    if not allow_missing:
        fallback_msg = " or its .csv fallback" if source.endswith(".parquet") else ""
        raise FileNotFoundError(f"File not found: {source}{fallback_msg}")

    return None


def load_model_any(source: str, s3, tmp_dir: Path) -> Path:
    if is_s3_uri(source):
        bucket, key = parse_s3_uri(source)
        dest = tmp_dir / key
        download_s3_object(s3, bucket, key, dest)
        return dest
    return Path(source)


def resolve_feature_columns(
    df: pd.DataFrame, allow_missing: bool = False
) -> Tuple[List[str], List[str], List[str], Dict[str, str]]:
    selected: List[str] = []
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    mapping: Dict[str, str] = {}
    missing: List[str] = []

    for spec in FEATURE_SPECS:
        candidates = (spec.canonical,) + spec.alternatives
        chosen = next((col for col in candidates if col in df.columns), None)
        if not chosen:
            missing.append(spec.canonical)
            continue
        selected.append(chosen)
        mapping[spec.canonical] = chosen
        if spec.kind == "categorical":
            categorical_cols.append(chosen)
        else:
            numeric_cols.append(chosen)

    if missing and not allow_missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Missing required feature columns: {missing_str}. "
            "Run preprocessing or pass --allow-missing to continue."
        )

    if not selected:
        raise ValueError("No feature columns found in dataset.")

    return selected, numeric_cols, categorical_cols, mapping


def coerce_feature_types(
    df: pd.DataFrame, numeric_cols: Iterable[str], categorical_cols: Iterable[str]
) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical_cols:
        df[col] = df[col].astype(str)
    return df


def build_pipeline(
    model_type: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    class_weight: str | None,
    args: argparse.Namespace,
    y_train: pd.Series,
) -> Pipeline:
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    if model_type == "logreg":
        classifier = LogisticRegression(
            max_iter=args.max_iter,
            C=args.C,
            class_weight=class_weight,
            solver="liblinear",
        )
    elif model_type == "rf":
        classifier = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight=class_weight,
            random_state=args.seed,
            n_jobs=-1,
        )
    elif model_type == "xgb":
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError:
            print(
                "xgboost is not installed. Install with:\n"
                "  pip install xgboost",
                file=sys.stderr,
            )
            raise SystemExit(2)

        pos = max(float((y_train == 1).sum()), 1.0)
        neg = max(float((y_train == 0).sum()), 1.0)
        scale_pos_weight = neg / pos
        classifier = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=args.seed,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> Dict[str, float]:
    probas = pipeline.predict_proba(X_test)[:, 1]
    preds = (probas >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, probas)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    order = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    for key in order:
        if key in metrics:
            value = metrics[key]
            formatted = "nan" if np.isnan(value) else f"{value:.4f}"
            print(f"{key:>10}: {formatted}")


def save_metadata(
    path: Path,
    model_type: str,
    feature_columns: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    mapping: Dict[str, str],
    metrics: Dict[str, float],
    train_rows: int,
    test_rows: int,
    threshold: float,
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "target": TARGET_COL,
        "features": {
            "selected": feature_columns,
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "canonical_map": mapping,
        },
        "metrics": metrics,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "threshold": threshold,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def train_command(args: argparse.Namespace) -> int:
    try:
        train_source, test_source = resolve_s3_sources(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    needs_s3 = is_s3_uri(train_source) or (test_source and is_s3_uri(test_source))
    s3 = build_s3_client(args.region, args.profile) if needs_s3 else None

    import tempfile

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            # train_source is required, so we don't allow missing
            train_df = load_csv_any(train_source, s3, tmp_dir, allow_missing=False)

            # test_source is optional
            test_df = (
                load_csv_any(test_source, s3, tmp_dir, allow_missing=True)
                if test_source
                else None
            )
            return train_command_from_frames(args, train_df, test_df)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: Could not load data. {exc}", file=sys.stderr)
        return 2



def train_command_from_frames(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
) -> int:
    if TARGET_COL not in train_df.columns:
        print(f"Missing target column: {TARGET_COL}", file=sys.stderr)
        return 2

    if test_df is None:
        print("Test dataset not found. Splitting training data automatically.")
        train_df, test_df = stratified_split(train_df, TARGET_COL, args.test_size, args.seed)

    if TARGET_COL not in test_df.columns:
        print(f"Missing target column in test data: {TARGET_COL}", file=sys.stderr)
        return 2

    try:
        feature_columns, numeric_cols, categorical_cols, mapping = resolve_feature_columns(
            train_df, allow_missing=args.allow_missing
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    train_df = coerce_feature_types(train_df, numeric_cols, categorical_cols)
    test_df = coerce_feature_types(test_df, numeric_cols, categorical_cols)

    X_train = train_df[feature_columns].copy()
    y_train = train_df[TARGET_COL].astype(int)
    X_test = test_df[feature_columns].copy()
    y_test = test_df[TARGET_COL].astype(int)

    class_weight = None if args.class_weight == "none" else "balanced"
    pipeline = build_pipeline(
        args.model_type, numeric_cols, categorical_cols, class_weight, args, y_train
    )
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test, args.threshold)
    print("\nMetrics:")
    print_metrics(metrics)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / args.model_name
    meta_path = model_dir / args.meta_name

    joblib.dump(pipeline, model_path)
    save_metadata(
        meta_path,
        args.model_type,
        feature_columns,
        numeric_cols,
        categorical_cols,
        mapping,
        metrics,
        len(train_df),
        len(test_df),
        args.threshold,
    )

    print(f"\nSaved model: {model_path}")
    print(f"Saved metadata: {meta_path}")
    if args.no_upload:
        return 0

    bucket = args.bucket
    if not bucket:
        print(
            "Missing S3 bucket for model upload. Set S3_BUCKET or pass --bucket, "
            "or use --no-upload to skip.",
            file=sys.stderr,
        )
        return 2

    prefix = (args.model_prefix or "").strip("/")
    model_key = f"{prefix}/{args.model_name}" if prefix else args.model_name
    meta_key = f"{prefix}/{args.meta_name}" if prefix else args.meta_name

    s3 = build_s3_client(args.region, args.profile)
    print(f"\nUploading model to s3://{bucket}/{model_key}")
    upload_s3_object(s3, model_path, bucket, model_key)
    print(f"Uploading metadata to s3://{bucket}/{meta_key}")
    upload_s3_object(s3, meta_path, bucket, meta_key)
    print(f"Uploaded model artifacts to s3://{bucket}/{prefix}/" if prefix else f"Uploaded model artifacts to s3://{bucket}/")
    return 0


def load_json_input(path: Path | None, row: str | None) -> pd.DataFrame:
    if path:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(row or "{}")

    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    raise ValueError("JSON input must be a dict or a list of dicts.")


def load_predict_input(args: argparse.Namespace) -> pd.DataFrame:
    sources = [args.input, args.json_path, args.row]
    if sum(1 for s in sources if s) != 1:
        raise ValueError("Provide exactly one of --input, --json, or --row.")

    if args.input:
        return load_csv(Path(args.input))
    return load_json_input(Path(args.json_path) if args.json_path else None, args.row)


def resolve_predict_sources(args: argparse.Namespace) -> argparse.Namespace:
    sources = [args.input, args.json_path, args.row]
    if sum(1 for s in sources if s) == 0:
        bucket = os.getenv("S3_BUCKET") or os.getenv("S3_Bucket")
        prefix = os.getenv("S3_PROCESSED_PREFIX", "processed")
        default_input = default_s3_uri(bucket, prefix, "test.parquet")
        if not default_input:
            raise ValueError("Provide --input/--json/--row or set S3_BUCKET.")
        args.input = default_input
    return args


def predict_command(args: argparse.Namespace) -> int:
    try:
        args = resolve_predict_sources(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    needs_s3 = any(
        is_s3_uri(value)
        for value in (args.model, args.meta, args.input, args.json_path)
        if value
    )
    s3 = build_s3_client(args.region, args.profile) if needs_s3 else None

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        model_path = load_model_any(args.model, s3, tmp_dir)
        if not model_path.exists():
            print(f"Model not found: {model_path}", file=sys.stderr)
            return 2

        pipeline = joblib.load(model_path)
        sources = [args.input, args.json_path, args.row]
        if sum(1 for s in sources if s) != 1:
            print("Provide exactly one of --input, --json, or --row.", file=sys.stderr)
            return 2
        try:
            if args.input and is_s3_uri(args.input):
                df = load_csv_any(args.input, s3, tmp_dir)
            elif args.json_path and is_s3_uri(args.json_path):
                json_path = load_model_any(args.json_path, s3, tmp_dir)
                df = load_json_input(json_path, None)
            else:
                df = load_predict_input(args)
        except (ValueError, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 2

        return predict_command_with_data(args, df, pipeline, model_path, s3, tmp_dir)


def predict_command_with_data(
    args: argparse.Namespace,
    df: pd.DataFrame,
    pipeline: Pipeline,
    model_path: Path,
    s3,
    tmp_dir: Path,
) -> int:
    meta = None
    meta_candidates: List[Path] = []

    if args.meta:
        meta_candidates.append(load_model_any(args.meta, s3, tmp_dir))
    else:
        meta_candidates.extend(
            [
                model_path.with_suffix(".json"),
                model_path.with_name(f"{model_path.stem}_meta.json"),
            ]
        )
        if is_s3_uri(args.model) and s3:
            bucket, key = parse_s3_uri(args.model)
            key_path = PurePosixPath(key)
            s3_candidates = [
                key_path.with_suffix(".json"),
                key_path.with_name(f"{key_path.stem}_meta.json"),
            ]
            for candidate in s3_candidates:
                dest = tmp_dir / str(candidate)
                if download_s3_object(s3, bucket, str(candidate), dest, allow_missing=True):
                    meta_candidates.insert(0, dest)
                    break

    for candidate in meta_candidates:
        if candidate.exists():
            try:
                meta = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = None
            break

    X = df
    if meta and "features" in meta and "selected" in meta["features"]:
        required = meta["features"]["selected"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"Missing columns required by the model: {', '.join(missing)}", file=sys.stderr)
            return 2
        numeric_cols = meta["features"].get("numeric", [])
        categorical_cols = meta["features"].get("categorical", [])
        X = coerce_feature_types(df[required], numeric_cols, categorical_cols)

    probas = pipeline.predict_proba(X)[:, 1]
    preds = (probas >= args.threshold).astype(int)

    output_df = df.copy()
    output_df["delay_probability"] = probas
    output_df["delay_prediction"] = preds

    if args.output:
        output_df.to_csv(args.output, index=False)
    else:
        sys.stdout.write(output_df.to_csv(index=False))
    return 0


def main() -> int:
    load_env_file()
    args = parse_args()
    if args.command == "train":
        return train_command(args)
    if args.command == "predict":
        return predict_command(args)
    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
