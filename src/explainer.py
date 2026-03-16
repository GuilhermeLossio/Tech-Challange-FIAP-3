#!/usr/bin/env python3
"""Compute SHAP values and export visualizations for trained models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import shap
except ModuleNotFoundError:
    print(
        "Missing dependency: shap. Install with:\n"
        "  pip install shap",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from model import (
    TARGET_COL,
    build_s3_client,
    coerce_feature_types,
    default_s3_uri,
    download_s3_object,
    is_s3_uri,
    load_env_file,
    load_csv_any,
    load_json_input,
    load_model_any,
    parse_s3_uri,
    upload_s3_object,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SHAP values and summary plots."
    )
    parser.add_argument("--model", default="models/delay_model.pkl", help="Path or s3:// URI to model.")
    parser.add_argument("--meta", default=None, help="Optional metadata JSON (path or s3://).")
    parser.add_argument(
        "--input",
        default=None,
        help="CSV input (path or s3://, default: S3 when S3_BUCKET is set).",
    )
    parser.add_argument("--json", dest="json_path", default=None, help="JSON input file.")
    parser.add_argument("--row", default=None, help="Inline JSON row.")
    parser.add_argument("--output-dir", default="models/explain", help="Output directory.")
    parser.add_argument("--max-rows", type=int, default=200, help="Max rows to explain.")
    parser.add_argument("--top-k", type=int, default=5, help="Top features per row.")
    parser.add_argument("--plot", action="store_true", help="Save SHAP summary plot.")
    parser.add_argument(
        "--bucket",
        default=os.getenv("S3_BUCKET") or os.getenv("S3_Bucket"),
        help="S3 bucket name for uploads (default: env S3_BUCKET).",
    )
    parser.add_argument(
        "--explain-prefix",
        default=os.getenv("S3_EXPLAIN_PREFIX", "explain"),
        help="S3 prefix for SHAP outputs (default: env S3_EXPLAIN_PREFIX or 'explain').",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Do not upload explain outputs to S3.",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS region (optional for S3 access).",
    )
    parser.add_argument(
        "--profile",
        default=os.getenv("AWS_PROFILE"),
        help="AWS profile (optional for S3 access).",
    )
    return parser.parse_args()


def load_input_frame(
    args: argparse.Namespace,
    s3,
    tmp_dir: Path,
) -> pd.DataFrame:
    sources = [args.input, args.json_path, args.row]
    if sum(1 for s in sources if s) == 0:
        bucket = os.getenv("S3_BUCKET") or os.getenv("S3_Bucket")
        prefix = os.getenv("S3_PROCESSED_PREFIX", "processed")
        default_input = default_s3_uri(bucket, prefix, "test.csv")
        if not default_input:
            raise ValueError("Provide --input/--json/--row or set S3_BUCKET.")
        args.input = default_input
        sources = [args.input, args.json_path, args.row]

    if sum(1 for s in sources if s) != 1:
        raise ValueError("Provide exactly one of --input, --json, or --row.")

    if args.input:
        return load_csv_any(args.input, s3, tmp_dir)

    if args.json_path:
        if is_s3_uri(args.json_path):
            json_path = load_model_any(args.json_path, s3, tmp_dir)
            return load_json_input(json_path, None)
        return load_json_input(Path(args.json_path), None)

    return load_json_input(None, args.row)


def load_metadata(
    args: argparse.Namespace,
    model_path: Path,
    s3,
    tmp_dir: Path,
) -> dict | None:
    meta = None
    candidates: List[Path] = []

    if args.meta:
        candidates.append(load_model_any(args.meta, s3, tmp_dir))
    else:
        candidates.extend(
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
                    candidates.insert(0, dest)
                    break

    for candidate in candidates:
        if candidate.exists():
            try:
                meta = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = None
            break
    return meta


def extract_pipeline_parts(pipeline) -> Tuple[object | None, object]:
    if hasattr(pipeline, "named_steps") and "preprocess" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps.get("model", pipeline)
        return preprocessor, model
    return None, pipeline


def get_feature_names(
    preprocessor,
    X_transformed: np.ndarray,
    raw_df: pd.DataFrame,
) -> List[str]:
    if preprocessor is None:
        return list(raw_df.columns)
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"f{i}" for i in range(X_transformed.shape[1])]


def densify(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def extract_shap_values(values: np.ndarray) -> np.ndarray:
    if values.ndim == 3:
        class_idx = 1 if values.shape[2] > 1 else 0
        return values[:, :, class_idx]
    return values


def build_top_factors(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: List[str],
    top_k: int,
) -> pd.DataFrame:
    rows = []
    for row_idx, row_vals in enumerate(shap_values):
        order = np.argsort(np.abs(row_vals))[::-1][:top_k]
        for rank, feat_idx in enumerate(order, start=1):
            rows.append(
                {
                    "row_index": row_idx,
                    "rank": rank,
                    "feature": feature_names[feat_idx],
                    "feature_value": feature_values[row_idx, feat_idx],
                    "shap_value": row_vals[feat_idx],
                    "abs_shap": abs(row_vals[feat_idx]),
                }
            )
    return pd.DataFrame(rows)


def upload_explain_outputs(
    s3,
    bucket: str,
    prefix: str,
    output_dir: Path,
) -> None:
    prefix = (prefix or "").strip("/")
    for path in output_dir.glob("*"):
        if path.is_dir():
            continue
        key = f"{prefix}/{path.name}" if prefix else path.name
        upload_s3_object(s3, path, bucket, key)


def main() -> int:
    load_env_file()
    args = parse_args()

    needs_s3 = any(
        is_s3_uri(value)
        for value in (args.model, args.meta, args.input, args.json_path)
        if value
    ) or (bool(args.bucket) and not args.no_upload)
    s3 = build_s3_client(args.region, args.profile) if needs_s3 else None

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        model_path = load_model_any(args.model, s3, tmp_dir)
        if not model_path.exists():
            print(f"Model not found: {model_path}", file=sys.stderr)
            return 2

        try:
            import joblib

            pipeline = joblib.load(model_path)
        except Exception as exc:
            print(f"Failed to load model: {exc}", file=sys.stderr)
            return 2

        try:
            df = load_input_frame(args, s3, tmp_dir)
        except (ValueError, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 2

        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])

        meta = load_metadata(args, model_path, s3, tmp_dir)
        if meta and "features" in meta and "selected" in meta["features"]:
            required = meta["features"]["selected"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                print(f"Missing columns required by the model: {', '.join(missing)}", file=sys.stderr)
                return 2
            numeric_cols = meta["features"].get("numeric", [])
            categorical_cols = meta["features"].get("categorical", [])
            df = coerce_feature_types(df[required], numeric_cols, categorical_cols)

        if args.max_rows and len(df) > args.max_rows:
            df = df.head(args.max_rows)

        preprocessor, model = extract_pipeline_parts(pipeline)
        X = df
        if preprocessor is not None:
            X = preprocessor.transform(df)
        X = np.asarray(densify(X))

        feature_names = get_feature_names(preprocessor, X, df)

        background = X if len(X) <= 100 else X[:100]
        explainer = shap.Explainer(model, background, feature_names=feature_names)
        shap_values = explainer(X)

        values = extract_shap_values(shap_values.values)
        if values.shape[1] != X.shape[1]:
            print("SHAP output shape does not match feature matrix.", file=sys.stderr)
            return 2

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        top_factors = build_top_factors(values, X, feature_names, args.top_k)
        top_factors.to_csv(output_dir / "shap_top_featured.csv", index=False)

        if args.plot:
            if plt is None:
                print("matplotlib not installed; cannot save plot.", file=sys.stderr)
            else:
                shap.summary_plot(values, X, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(output_dir / "shap_summary.png", dpi=160)
                plt.close()

        print(f"Saved SHAP outputs to: {output_dir}")
        if args.no_upload:
            return 0

        if not args.bucket:
            print(
                "Missing S3 bucket for explain upload. Set S3_BUCKET or pass --bucket, "
                "or use --no-upload to skip.",
                file=sys.stderr,
            )
            return 2

        if s3 is None:
            s3 = build_s3_client(args.region, args.profile)

        upload_explain_outputs(s3, args.bucket, args.explain_prefix, output_dir)
        prefix = (args.explain_prefix or "").strip("/")
        if prefix:
            print(f"Uploaded SHAP outputs to s3://{args.bucket}/{prefix}/")
        else:
            print(f"Uploaded SHAP outputs to s3://{args.bucket}/")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
