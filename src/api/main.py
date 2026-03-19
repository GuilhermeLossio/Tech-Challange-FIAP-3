#!/usr/bin/env python3
"""FastAPI endpoints for Flight Advisor."""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Ensure src/ is on the import path when running via uvicorn
SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model import (  # noqa: E402
    FEATURE_SPECS,
    TARGET_COL,
    build_s3_client,
    coerce_feature_types,
    default_s3_uri,
    is_s3_uri,
    load_env_file,
    load_model_any,
    parse_s3_uri,
)


app = FastAPI(title="Flight Advisor API", version="0.1.0")

origins = os.getenv("CORS_ORIGINS")
if origins:
    allowed = [origin.strip() for origin in origins.split(",") if origin.strip()]
else:
    allowed = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_env_file()

def mount_dash_app(fastapi_app: FastAPI) -> None:
    if os.getenv("ENABLE_DASH", "1") != "1":
        return
    try:
        from starlette.middleware.wsgi import WSGIMiddleware
        from dashboard.app import create_dash_app
    except Exception as exc:
        print(f"Dash not mounted: {exc}", file=sys.stderr)
        return
    dash_app = create_dash_app("/dashboard/", "/")
    fastapi_app.mount("/dashboard", WSGIMiddleware(dash_app.server))


mount_dash_app(app)


class AdviseRequest(BaseModel):
    origin_airport: str = Field(..., description="Origin IATA code")
    destination_airport: str = Field(..., description="Destination IATA code")
    airline: str = Field(..., description="Airline IATA code")
    scheduled_departure: int = Field(..., description="Scheduled departure time (HHMM)")
    month: int | None = Field(None, ge=1, le=12)
    day_of_week: int | None = Field(None, ge=1, le=7)
    day: int | None = Field(None, ge=1, le=31)
    year: int | None = Field(None, ge=2000, le=2100)
    flight_date: str | None = Field(None, description="YYYY-MM-DD")
    distance: float | None = Field(None, description="Route distance in miles")
    question: str | None = Field(None, description="Free-form user question")

    @validator("origin_airport", "destination_airport", "airline")
    def normalize_codes(cls, value: str) -> str:
        return value.strip().upper()


class Factor(BaseModel):
    feature: str
    impact: str


class AdviseResponse(BaseModel):
    delay_probability: float
    risk_level: str
    top_factors: List[Factor]
    advice: str


RATE_COLS = [
    "ORIGIN_DELAY_RATE",
    "DEST_DELAY_RATE",
    "CARRIER_DELAY_RATE",
    "ROUTE_DELAY_RATE",
    "CARRIER_DELAY_RATE_DOW",
]


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def time_of_day(hour: int) -> str:
    if 5 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 16:
        return "afternoon"
    if 17 <= hour <= 21:
        return "evening"
    return "night"


def build_holiday_set(dates: List[pd.Timestamp]) -> set[pd.Timestamp]:
    from pandas.tseries.holiday import USFederalHolidayCalendar

    if not dates:
        return set()
    start = min(dates).normalize()
    end = max(dates).normalize()
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start, end=end)
    return set(pd.to_datetime(holidays).normalize())


def infer_date(req: AdviseRequest) -> date:
    if req.flight_date:
        return date.fromisoformat(req.flight_date)

    if req.year and req.month and req.day:
        return date(req.year, req.month, req.day)

    if req.month is None or req.day_of_week is None:
        raise ValueError("Provide flight_date or (month + day_of_week).")

    today = date.today()
    year = req.year or today.year
    if req.month < today.month:
        year += 1

    first = date(year, req.month, 1)
    target = req.day_of_week
    offset = (target - (first.weekday() + 1)) % 7
    return first + timedelta(days=offset)


def build_route_distance_map(df: pd.DataFrame) -> Tuple[Dict[str, float], float]:
    if "ROUTE" not in df.columns:
        df = df.copy()
        df["ROUTE"] = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)
    distance_map = df.groupby("ROUTE")["DISTANCE"].mean().to_dict()
    global_dist = float(df["DISTANCE"].mean())
    return distance_map, global_dist


def build_rate_maps(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], float]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Rates dataset must include {TARGET_COL}.")

    global_rate = float(df[TARGET_COL].mean())
    if "ROUTE" not in df.columns:
        df = df.copy()
        df["ROUTE"] = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)
    if "AIRLINE_DOW" not in df.columns:
        df = df.copy()
        df["AIRLINE_DOW"] = df["AIRLINE"].astype(str) + "_" + df["DAY_OF_WEEK"].astype(str)

    maps = {
        "ORIGIN_DELAY_RATE": df.groupby("ORIGIN_AIRPORT")[TARGET_COL].mean().to_dict(),
        "DEST_DELAY_RATE": df.groupby("DESTINATION_AIRPORT")[TARGET_COL].mean().to_dict(),
        "CARRIER_DELAY_RATE": df.groupby("AIRLINE")[TARGET_COL].mean().to_dict(),
        "ROUTE_DELAY_RATE": df.groupby("ROUTE")[TARGET_COL].mean().to_dict(),
        "CARRIER_DELAY_RATE_DOW": df.groupby("AIRLINE_DOW")[TARGET_COL].mean().to_dict(),
    }
    return maps, global_rate


def apply_rate_maps(
    df: pd.DataFrame,
    maps: Dict[str, Dict[str, float]],
    global_rate: float,
) -> pd.DataFrame:
    df = df.copy()
    df["ORIGIN_DELAY_RATE"] = df["ORIGIN_AIRPORT"].map(maps["ORIGIN_DELAY_RATE"]).fillna(global_rate)
    df["DEST_DELAY_RATE"] = df["DESTINATION_AIRPORT"].map(maps["DEST_DELAY_RATE"]).fillna(global_rate)
    df["CARRIER_DELAY_RATE"] = df["AIRLINE"].map(maps["CARRIER_DELAY_RATE"]).fillna(global_rate)
    df["ROUTE_DELAY_RATE"] = df["ROUTE"].map(maps["ROUTE_DELAY_RATE"]).fillna(global_rate)
    df["CARRIER_DELAY_RATE_DOW"] = df["AIRLINE_DOW"].map(maps["CARRIER_DELAY_RATE_DOW"]).fillna(global_rate)
    df["ROTA_DELAY_RATE"] = df["ROUTE_DELAY_RATE"]
    return df


def build_features(
    req: AdviseRequest,
    maps: Dict[str, Dict[str, float]],
    global_rate: float,
    route_distance_map: Dict[str, float],
    global_distance: float,
) -> pd.DataFrame:
    flight_date = infer_date(req)
    day_of_week = req.day_of_week or (flight_date.weekday() + 1)

    route = f"{req.origin_airport}_{req.destination_airport}"
    distance = req.distance or route_distance_map.get(route) or global_distance
    scheduled = int(req.scheduled_departure)

    row = {
        "YEAR": flight_date.year,
        "MONTH": req.month or flight_date.month,
        "DAY": flight_date.day,
        "DAY_OF_WEEK": day_of_week,
        "SCHEDULED_DEPARTURE": scheduled,
        "DISTANCE": distance,
        "ORIGIN_AIRPORT": req.origin_airport,
        "DESTINATION_AIRPORT": req.destination_airport,
        "AIRLINE": req.airline,
    }
    df = pd.DataFrame([row])

    hours = (df["SCHEDULED_DEPARTURE"].fillna(0).astype(int) // 100).clip(0, 23)
    df["TIME_OF_DAY"] = hours.map(time_of_day)
    df["SEASON"] = df["MONTH"].astype(int).map(season_from_month)

    holiday_set = build_holiday_set([pd.Timestamp(flight_date)])
    df["IS_HOLIDAY"] = pd.to_datetime([flight_date]).normalize().isin(holiday_set).astype(int)

    df["ROUTE"] = route
    df["AIRLINE_DOW"] = df["AIRLINE"].astype(str) + "_" + df["DAY_OF_WEEK"].astype(str)

    df["PERIODO_DIA"] = df["TIME_OF_DAY"]
    df["ESTACAO"] = df["SEASON"]
    df["IS_FERIADO"] = df["IS_HOLIDAY"]
    df["ROTA"] = df["ROUTE"]

    df = apply_rate_maps(df, maps, global_rate)
    return df


def compute_top_factors(
    df: pd.DataFrame,
    global_rate: float,
    top_k: int = 3,
) -> List[Factor]:
    factors = []
    for col in RATE_COLS:
        rate = float(df[col].iloc[0])
        delta = rate - global_rate
        impact = f"{delta * 100:+.1f}%"
        factors.append((col.lower(), abs(delta), impact))

    factors.sort(key=lambda item: item[1], reverse=True)
    return [Factor(feature=name, impact=impact) for name, _, impact in factors[:top_k]]


def risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    if prob >= 0.4:
        return "MEDIUM"
    return "LOW"


def advice_text(prob: float, level: str, top_factors: List[Factor]) -> str:
    base = f"This flight has a {level} risk of delay ({prob:.0%})."
    if top_factors:
        best = top_factors[0]
        return f"{base} The strongest signal is {best.feature} ({best.impact})."
    return base


def resolve_uri(env_key: str, default_uri: str | None) -> str | None:
    value = os.getenv(env_key)
    return value or default_uri


@lru_cache
def load_assets():
    load_env_file()

    bucket = os.getenv("S3_BUCKET") or os.getenv("S3_Bucket")
    processed_prefix = os.getenv("S3_PROCESSED_PREFIX", "processed")
    model_prefix = os.getenv("S3_MODEL_PREFIX", "models")

    model_default = default_s3_uri(bucket, model_prefix, "delay_model.pkl") if bucket else "models/delay_model.pkl"
    meta_default = default_s3_uri(bucket, model_prefix, "delay_model_meta.json") if bucket else "models/delay_model_meta.json"
    rates_default = default_s3_uri(bucket, processed_prefix, "train.parquet") if bucket else "data/processed/train.parquet"

    model_uri = resolve_uri("MODEL_URI", model_default)
    meta_uri = resolve_uri("MODEL_META_URI", meta_default)
    rates_uri = resolve_uri("RATES_SOURCE", rates_default)

    needs_s3 = any(is_s3_uri(value) for value in (model_uri, meta_uri, rates_uri) if value)
    s3 = build_s3_client(os.getenv("AWS_REGION", "us-east-1"), os.getenv("AWS_PROFILE")) if needs_s3 else None

    import tempfile
    import joblib

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        model_path = load_model_any(model_uri, s3, tmp_dir)
        pipeline = joblib.load(model_path)

        meta = None
        if meta_uri:
            meta_path = load_model_any(meta_uri, s3, tmp_dir)
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    meta = None

        if not rates_uri:
            raise RuntimeError("Missing RATES_SOURCE or S3_BUCKET for rate maps.")

        rates_path = load_model_any(rates_uri, s3, tmp_dir)
        usecols = [
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "AIRLINE",
            "DAY_OF_WEEK",
            "DISTANCE",
            TARGET_COL,
        ]
        if rates_path.suffix.lower() == ".parquet":
            rates_df = pd.read_parquet(rates_path, columns=usecols)
        else:
            rates_df = pd.read_csv(rates_path, usecols=usecols)

    route_distance_map, global_distance = build_route_distance_map(rates_df)
    rate_maps, global_rate = build_rate_maps(rates_df)

    return pipeline, meta, rate_maps, global_rate, route_distance_map, global_distance


@app.get("/health")
def health():
    try:
        _ = load_assets()
        return {"status": "ok"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/advise", response_model=AdviseResponse)
def advise(payload: AdviseRequest):
    try:
        pipeline, meta, maps, global_rate, route_distance_map, global_distance = load_assets()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        df = build_features(payload, maps, global_rate, route_distance_map, global_distance)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    X = df
    if meta and "features" in meta and "selected" in meta["features"]:
        required = meta["features"]["selected"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns required by the model: {', '.join(missing)}",
            )
        numeric_cols = meta["features"].get("numeric", [])
        categorical_cols = meta["features"].get("categorical", [])
        X = coerce_feature_types(df[required], numeric_cols, categorical_cols)

    prob = float(pipeline.predict_proba(X)[:, 1][0])
    level = risk_level(prob)
    top = compute_top_factors(df, global_rate)
    advice = advice_text(prob, level, top)
    return AdviseResponse(
        delay_probability=prob,
        risk_level=level,
        top_factors=top,
        advice=advice,
    )
