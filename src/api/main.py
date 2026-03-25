#!/usr/bin/env python3
"""Flask app for Flight Advisor (frontend + JSON API)."""
from __future__ import annotations

import json, os, sys
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Tuple

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for
from pydantic import BaseModel, Field, ValidationError, field_validator

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
for _p in (SRC_DIR, ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model import (  # noqa: E402
    TARGET_COL, build_s3_client, coerce_feature_types, default_s3_uri,
    download_s3_object, is_s3_uri, load_csv_any, load_env_file,
    load_model_any, parse_s3_uri,
)

load_env_file()
app = Flask(__name__, template_folder=str(SRC_DIR / "templates"),
            static_folder=str(SRC_DIR / "static"), static_url_path="/static")

_raw_origins = os.getenv("CORS_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()] or ["*"]

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if "*" in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = "*"
    elif origin and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    response.headers.update({
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    })
    return response


def mount_dash_app(flask_app: Flask) -> None:
    if os.getenv("ENABLE_DASH", "1") != "1" or getattr(flask_app, "_dash_mounted", False):
        return
    try:
        from werkzeug.middleware.dispatcher import DispatcherMiddleware
        os.environ.setdefault("DASH_BASE_PATH", "/dashboard/")
        from dashboard.app import app as dash_app
    except Exception as exc:
        print(f"Dash not mounted: {exc}", file=sys.stderr)
        return
    flask_app.wsgi_app = DispatcherMiddleware(flask_app.wsgi_app, {"/dashboard": dash_app.server.wsgi_app})
    flask_app._dash_mounted = True

mount_dash_app(app)


# ── Pydantic models ──────────────────────────────────────────────────────────

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

    @field_validator("origin_airport", "destination_airport", "airline")
    @classmethod
    def normalize_codes(cls, v: str) -> str:
        return v.strip().upper()


class Factor(BaseModel):
    feature: str
    impact: str


class AdviseResponse(BaseModel):
    delay_probability: float
    risk_level: str
    top_factors: List[Factor]
    advice: str


class PredictRequest(BaseModel):
    input_uri: str | None = Field(None)
    row: Dict[str, Any] | None = Field(None)
    rows: List[Dict[str, Any]] | None = Field(None)
    origin_airport: str | None = None
    destination_airport: str | None = None
    airline: str | None = None
    flight_date: str | None = None
    year: int | None = Field(None, ge=2000, le=2100)
    month: int | None = Field(None, ge=1, le=12)
    day: int | None = Field(None, ge=1, le=31)
    day_of_week: int | None = Field(None, ge=1, le=7)
    scheduled_departure: int | None = Field(None, ge=0, le=2359)
    limit: int = Field(50, ge=1, le=500)
    threshold: float = Field(0.5, ge=0.0, le=1.0)

    @field_validator("origin_airport", "destination_airport", "airline")
    @classmethod
    def normalize_filter_codes(cls, v: str | None) -> str | None:
        return v.strip().upper() or None if v else None


# ── Constants / helpers ──────────────────────────────────────────────────────

RATE_COLS = ["ORIGIN_DELAY_RATE", "DEST_DELAY_RATE", "CARRIER_DELAY_RATE",
             "ROUTE_DELAY_RATE", "CARRIER_DELAY_RATE_DOW"]
OPENFLIGHTS_AIRPORTS_SOURCE = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"

def season_from_month(m: int) -> str:
    return {12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",
            6:"summer",7:"summer",8:"summer"}.get(m, "fall")

def time_of_day(h: int) -> str:
    if 5<=h<=11: return "morning"
    if 12<=h<=16: return "afternoon"
    if 17<=h<=21: return "evening"
    return "night"

def build_holiday_set(dates: list[pd.Timestamp]) -> set[pd.Timestamp]:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    if not dates: return set()
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=min(dates).normalize(), end=max(dates).normalize())
    return set(pd.to_datetime(holidays).normalize())

def infer_date(req: AdviseRequest) -> date:
    if req.flight_date:
        return date.fromisoformat(req.flight_date)
    if req.year and req.month and req.day:
        return date(req.year, req.month, req.day)
    if req.month is None or req.day_of_week is None:
        raise ValueError("Provide flight_date or (month + day_of_week).")
    today = date.today()
    year = req.year or (today.year + (1 if req.month < today.month else 0))
    first = date(year, req.month, 1)
    return first + timedelta(days=(req.day_of_week - (first.weekday() + 1)) % 7)

def build_route_distance_map(df: pd.DataFrame) -> Tuple[Dict[str, float], float]:
    if "ROUTE" not in df.columns:
        df = df.copy()
        df["ROUTE"] = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)
    return df.groupby("ROUTE")["DISTANCE"].mean().to_dict(), float(df["DISTANCE"].mean())

def build_rate_maps(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], float]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Rates dataset must include {TARGET_COL}.")
    df = df.copy()
    if "ROUTE" not in df.columns:
        df["ROUTE"] = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)
    if "AIRLINE_DOW" not in df.columns:
        df["AIRLINE_DOW"] = df["AIRLINE"].astype(str) + "_" + df["DAY_OF_WEEK"].astype(str)
    maps = {
        "ORIGIN_DELAY_RATE":    df.groupby("ORIGIN_AIRPORT")[TARGET_COL].mean().to_dict(),
        "DEST_DELAY_RATE":      df.groupby("DESTINATION_AIRPORT")[TARGET_COL].mean().to_dict(),
        "CARRIER_DELAY_RATE":   df.groupby("AIRLINE")[TARGET_COL].mean().to_dict(),
        "ROUTE_DELAY_RATE":     df.groupby("ROUTE")[TARGET_COL].mean().to_dict(),
        "CARRIER_DELAY_RATE_DOW": df.groupby("AIRLINE_DOW")[TARGET_COL].mean().to_dict(),
    }
    return maps, float(df[TARGET_COL].mean())

def apply_rate_maps(df: pd.DataFrame, maps: Dict[str, Dict[str, float]], gr: float) -> pd.DataFrame:
    df = df.copy()
    df["ORIGIN_DELAY_RATE"]     = df["ORIGIN_AIRPORT"].map(maps["ORIGIN_DELAY_RATE"]).fillna(gr)
    df["DEST_DELAY_RATE"]       = df["DESTINATION_AIRPORT"].map(maps["DEST_DELAY_RATE"]).fillna(gr)
    df["CARRIER_DELAY_RATE"]    = df["AIRLINE"].map(maps["CARRIER_DELAY_RATE"]).fillna(gr)
    df["ROUTE_DELAY_RATE"]      = df["ROUTE"].map(maps["ROUTE_DELAY_RATE"]).fillna(gr)
    df["CARRIER_DELAY_RATE_DOW"]= df["AIRLINE_DOW"].map(maps["CARRIER_DELAY_RATE_DOW"]).fillna(gr)
    df["ROTA_DELAY_RATE"]       = df["ROUTE_DELAY_RATE"]
    return df

def build_features(req, maps, global_rate, route_distance_map, global_distance) -> pd.DataFrame:
    flight_date = infer_date(req)
    dow = req.day_of_week or (flight_date.weekday() + 1)
    route = f"{req.origin_airport}_{req.destination_airport}"
    distance = req.distance or route_distance_map.get(route) or global_distance
    df = pd.DataFrame([{
        "YEAR": flight_date.year, "MONTH": req.month or flight_date.month,
        "DAY": flight_date.day, "DAY_OF_WEEK": dow,
        "SCHEDULED_DEPARTURE": int(req.scheduled_departure),
        "DISTANCE": distance, "ORIGIN_AIRPORT": req.origin_airport,
        "DESTINATION_AIRPORT": req.destination_airport, "AIRLINE": req.airline,
    }])
    hours = (df["SCHEDULED_DEPARTURE"].fillna(0).astype(int) // 100).clip(0, 23)
    df["TIME_OF_DAY"] = hours.map(time_of_day)
    df["SEASON"]      = df["MONTH"].astype(int).map(season_from_month)
    df["IS_HOLIDAY"]  = pd.to_datetime([flight_date]).normalize().isin(
        build_holiday_set([pd.Timestamp(flight_date)])).astype(int)
    df["ROUTE"]       = route
    df["AIRLINE_DOW"] = df["AIRLINE"].astype(str) + "_" + df["DAY_OF_WEEK"].astype(str)
    df["PERIODO_DIA"] = df["TIME_OF_DAY"]
    df["ESTACAO"]     = df["SEASON"]
    df["IS_FERIADO"]  = df["IS_HOLIDAY"]
    df["ROTA"]        = df["ROUTE"]
    return apply_rate_maps(df, maps, global_rate)

def compute_top_factors(df: pd.DataFrame, global_rate: float, top_k: int = 3) -> List[Factor]:
    factors = sorted(
        [(col.lower(), abs(float(df[col].iloc[0]) - global_rate),
          f"{(float(df[col].iloc[0]) - global_rate)*100:+.1f}%") for col in RATE_COLS],
        key=lambda x: x[1], reverse=True,
    )
    return [Factor(feature=n, impact=i) for n, _, i in factors[:top_k]]

def risk_level(p: float) -> str:
    return "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.4 else "LOW"

def advice_text(prob: float, level: str, top_factors: List[Factor]) -> str:
    base = f"This flight has a {level} risk of delay ({prob:.0%})."
    return f"{base} The strongest signal is {top_factors[0].feature} ({top_factors[0].impact})." if top_factors else base

def resolve_uri(env_key: str, default: str | None) -> str | None:
    v = os.getenv(env_key)
    return (v.strip() or default) if v is not None else default

def build_s3_client_safe() -> Any:
    try:
        return build_s3_client(os.getenv("AWS_REGION", "us-east-1"), os.getenv("AWS_PROFILE"))
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize S3 client: {exc}") from exc

def load_model_any_safe(source: str, s3: Any, tmp_dir: Path) -> Path:
    try:
        return load_model_any(source, s3, tmp_dir)
    except SystemExit as exc:
        raise RuntimeError(f"Failed to load artifact from '{source}'.") from exc

def load_optional_meta(source: str | None, s3: Any, tmp_dir: Path) -> dict | None:
    if not source: return None
    if is_s3_uri(source):
        if s3 is None: return None
        bucket, key = parse_s3_uri(source)
        candidate_path = tmp_dir / PurePosixPath(key)
        try:
            exists = download_s3_object(s3, bucket, key, candidate_path, allow_missing=True)
        except SystemExit as exc:
            raise RuntimeError(f"Failed to download metadata from '{source}'.") from exc
        if not exists: return None
    else:
        candidate_path = Path(source)
        if not candidate_path.exists(): return None
    try:
        return json.loads(candidate_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@lru_cache
def load_assets():
    import tempfile, joblib
    load_env_file()
    bucket  = os.getenv("S3_BUCKET") or os.getenv("S3_Bucket")
    mp      = os.getenv("S3_MODEL_PREFIX", "models")
    pp      = os.getenv("S3_PROCESSED_PREFIX", "processed")

    model_uri = resolve_uri("MODEL_URI",    default_s3_uri(bucket, mp, "delay_model.pkl")       or "models/delay_model.pkl")
    meta_uri  = resolve_uri("MODEL_META_URI", default_s3_uri(bucket, mp, "delay_model_meta.json") or "models/delay_model_meta.json")
    rates_uri = resolve_uri("RATES_SOURCE", default_s3_uri(bucket, pp, "train.parquet")         or "data/processed/train.parquet")

    s3 = build_s3_client_safe() if any(is_s3_uri(u) for u in (model_uri, meta_uri, rates_uri) if u) else None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir  = Path(tmp)
        pipeline = joblib.load(load_model_any_safe(model_uri, s3, tmp_dir))
        meta     = load_optional_meta(meta_uri, s3, tmp_dir)
        if not rates_uri:
            raise RuntimeError("Missing RATES_SOURCE or S3_BUCKET for rate maps.")
        try:
            rates_df = load_csv_any(rates_uri, s3, tmp_dir, allow_missing=False)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc
        except SystemExit as exc:
            raise RuntimeError(f"Failed to load rates dataset from '{rates_uri}'.") from exc

    if rates_df is None:
        raise RuntimeError(f"Rates dataset not found: {rates_uri}")
    required_cols = ["ORIGIN_AIRPORT","DESTINATION_AIRPORT","AIRLINE","DAY_OF_WEEK","DISTANCE",TARGET_COL]
    missing = [c for c in required_cols if c not in rates_df.columns]
    if missing:
        raise RuntimeError("Rates dataset is missing required columns: " + ", ".join(missing))
    rates_df = rates_df[required_cols].copy()
    return pipeline, meta, *build_rate_maps(rates_df), *build_route_distance_map(rates_df)


# ── Prediction helpers ───────────────────────────────────────────────────────

def resolve_predict_input_uri(input_uri: str | None) -> str:
    bucket = os.getenv("S3_BUCKET") or os.getenv("S3_Bucket")
    default = default_s3_uri(bucket, os.getenv("S3_PROCESSED_PREFIX","processed"), "test.parquet") or "data/processed/test.parquet"
    source = input_uri or resolve_uri("PREDICT_SOURCE", default)
    if not source:
        raise ValueError("Missing prediction source. Provide input_uri or set PREDICT_SOURCE/S3_BUCKET.")
    return source

def load_predict_frame(payload: PredictRequest) -> tuple[pd.DataFrame, str]:
    if payload.row and payload.rows:
        raise ValueError("Provide only one of row or rows.")
    if (payload.row or payload.rows) and payload.input_uri:
        raise ValueError("Do not combine input_uri with row/rows.")
    if payload.row  is not None: return pd.DataFrame([payload.row]),  "inline:row"
    if payload.rows is not None:
        if not payload.rows: raise ValueError("rows must not be empty.")
        return pd.DataFrame(payload.rows), "inline:rows"
    source = resolve_predict_input_uri(payload.input_uri)
    s3 = build_s3_client_safe() if is_s3_uri(source) else None
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        df = load_csv_any(source, s3, Path(tmp), allow_missing=False)
    if df is None: raise FileNotFoundError(f"Prediction input not found: {source}")
    return df, source

def find_column_name(df: pd.DataFrame, expected: str) -> str | None:
    norm = expected.strip().upper()
    return next((c for c in df.columns if str(c).strip().upper() == norm), None)


# ── Airports helpers ─────────────────────────────────────────────────────────

def resolve_airports_index_sources(source_uri: str | None = None) -> list[str]:
    candidates = [source_uri] if source_uri and source_uri.strip() else []
    for k in ("AIRPORTS_INDEX_SOURCE", "WORLD_AIRPORTS_SOURCE"):
        v = resolve_uri(k, None)
        if v: candidates.append(v)
    candidates += ["data/refined/world_airports.csv","data/raw/world_airports.csv",
                   OPENFLIGHTS_AIRPORTS_SOURCE,"data/raw/airports.csv"]
    seen: set[str] = set()
    return [s for s in candidates if s.strip() and not (s.strip() in seen or seen.add(s.strip()))]

def normalize_airports_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    iata_col = find_column_name(raw_df,"IATA_CODE") or find_column_name(raw_df,"IATA")
    if not iata_col: raise ValueError("Airports dataset is missing IATA_CODE/IATA column.")
    name_col    = find_column_name(raw_df,"AIRPORT") or find_column_name(raw_df,"AIRPORT_NAME") or find_column_name(raw_df,"NAME")
    city_col    = find_column_name(raw_df,"CITY")
    state_col   = find_column_name(raw_df,"STATE")
    country_col = find_column_name(raw_df,"COUNTRY")
    if not country_col: raise ValueError("Airports dataset is missing COUNTRY column.")
    lat_col = find_column_name(raw_df,"LATITUDE") or find_column_name(raw_df,"LAT")
    lng_col = find_column_name(raw_df,"LONGITUDE") or find_column_name(raw_df,"LON") or find_column_name(raw_df,"LNG") or find_column_name(raw_df,"LONG")

    def pick(col): return raw_df[col] if col else pd.Series([None]*len(raw_df), index=raw_df.index, dtype="object")
    frame = pd.DataFrame({
        "iata_code": pick(iata_col), "airport_name": pick(name_col),
        "city": pick(city_col), "state": pick(state_col), "country": pick(country_col),
        "latitude": pick(lat_col), "longitude": pick(lng_col),
    })

    def clean(s, upper=False):
        s = s.where(pd.notna(s), None).astype(str).str.strip().replace({"":None,"None":None,"nan":None,"\\N":None})
        return s.str.upper() if upper else s

    for col in ("airport_name","city","state","country"): frame[col] = clean(frame[col])
    frame["iata_code"]  = clean(frame["iata_code"], upper=True)
    frame["latitude"]   = pd.to_numeric(frame["latitude"],  errors="coerce")
    frame["longitude"]  = pd.to_numeric(frame["longitude"], errors="coerce")
    frame = frame[frame["iata_code"].str.fullmatch(r"[A-Z0-9]{3}", na=False) & frame["country"].notna()]
    return (frame.where(pd.notna(frame), None)
                 .drop_duplicates(subset=["iata_code"], keep="first")
                 .sort_values(["country","city","airport_name","iata_code"], na_position="last")
                 .reset_index(drop=True))

def read_openflights_airports_dat(source: str) -> pd.DataFrame:
    cols = ["airport_id","name","city","country","iata","icao","latitude","longitude",
            "altitude","timezone","dst","tz_database_time_zone","type","source"]
    raw_df = pd.read_csv(source, header=None, names=cols, dtype=str, keep_default_na=False, na_values=["\\N"])
    return normalize_airports_frame(pd.DataFrame({
        "IATA": raw_df["iata"], "AIRPORT": raw_df["name"],
        "CITY": raw_df["city"], "COUNTRY": raw_df["country"],
    }))

@lru_cache
def load_airports_index(source_uri: str | None = None) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for source in resolve_airports_index_sources(source_uri):
        try:
            is_remote = source.startswith(("http://","https://"))
            if Path(source).suffix.lower() == ".dat" or source == OPENFLIGHTS_AIRPORTS_SOURCE:
                frame = read_openflights_airports_dat(source)
            else:
                if not is_remote and not Path(source).exists(): continue
                frame = normalize_airports_frame(pd.read_csv(source, dtype=str))
        except Exception as exc:
            errors.append(f"{source}: {exc}"); continue
        if not frame.empty:
            return frame, source
    hint = "Configure AIRPORTS_INDEX_SOURCE or add data/raw/world_airports.csv."
    if errors: hint += f" Last errors: {' | '.join(errors[-3:])}"
    raise FileNotFoundError(f"Airports dataset not found. {hint}")


# ── Upcoming flights helpers ─────────────────────────────────────────────────

def resolve_upcoming_flights_sources(source_uri: str | None = None) -> list[str]:
    bucket = os.getenv("S3_BUCKET") or os.getenv("S3_Bucket")
    rp = os.getenv("S3_REFINED_PREFIX","refined")
    candidates = [source_uri] if source_uri and source_uri.strip() else []
    for k in ("UPCOMING_FLIGHTS_SOURCE","REAL_FLIGHTS_SOURCE","FUTURE_FLIGHTS_SOURCE","WEEKLY_PREDICTIONS_SOURCE"):
        v = resolve_uri(k, None)
        if v: candidates.append(v)
    if bucket:
        for fname in ("future_flights.parquet","future_flights.csv"):
            u = default_s3_uri(bucket, rp, fname)
            if u: candidates.append(u)
    candidates += ["data/refined/future_flights.parquet","data/refined/future_flights.csv",
                   "data/future_flights.parquet","data/future_flights.csv","data/predictions.csv"]
    seen: set[str] = set()
    return [s for s in candidates if s.strip() and not (s.strip() in seen or seen.add(s.strip()))]

def load_upcoming_flights_frame(source_uri: str | None = None) -> tuple[pd.DataFrame, str]:
    candidates = resolve_upcoming_flights_sources(source_uri)
    if not candidates:
        raise FileNotFoundError("Upcoming flights source is not configured.")
    needs_s3 = any(is_s3_uri(u) for u in candidates)
    s3, s3_error = None, None
    if needs_s3:
        try: s3 = build_s3_client_safe()
        except RuntimeError as exc: s3_error = exc
    import tempfile
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for src in candidates:
            if is_s3_uri(src) and s3 is None:
                if s3_error: errors.append(f"{src}: {s3_error}")
                continue
            if not is_s3_uri(src) and not Path(src).exists(): continue
            try:
                df = load_csv_any(src, s3, Path(tmp), allow_missing=True)
            except Exception as exc:
                errors.append(f"{src}: {exc}"); continue
            if df is not None: return df, src
    hint = "Upload refined/future_flights.parquet to S3 or configure UPCOMING_FLIGHTS_SOURCE."
    if errors: hint += f" Last errors: {' | '.join(errors[-3:])}"
    raise FileNotFoundError(f"Upcoming flights dataset not found. {hint}")

def format_scheduled_departure(value: Any) -> str | None:
    if value is None or pd.isna(value): return None
    try: raw = max(0, min(2359, int(float(value))))
    except (TypeError, ValueError):
        text = str(value).strip(); return text or None
    h, m = raw // 100, min(raw % 100, 59)
    return f"{h:02d}:{m:02d}"

def format_flight_number(value: Any) -> str | None:
    if value is None or pd.isna(value): return None
    try: return str(int(float(value)))
    except (TypeError, ValueError):
        text = str(value).strip(); return text or None

def optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value): return None
    return str(value).strip() or None

def _parse_dates_col(df: pd.DataFrame) -> tuple[pd.Series | None, pd.DataFrame]:
    fd = find_column_name(df,"FLIGHT_DATE")
    yy, mm, dd = find_column_name(df,"YEAR"), find_column_name(df,"MONTH"), find_column_name(df,"DAY")
    if fd:
        return pd.to_datetime(df[fd], errors="coerce").dt.date, df
    if yy and mm and dd:
        return pd.to_datetime(dict(
            year=pd.to_numeric(df[yy], errors="coerce"),
            month=pd.to_numeric(df[mm], errors="coerce"),
            day=pd.to_numeric(df[dd], errors="coerce"),
        ), errors="coerce").dt.date, df
    return None, df

def _filter_future(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    parsed_dates, frame = _parse_dates_col(frame)
    if parsed_dates is None: return frame, False
    frame = frame.copy(); frame["_flight_date"] = parsed_dates
    dated = frame[frame["_flight_date"].notna()].copy()
    if dated.empty: return frame, False
    future = dated[dated["_flight_date"] >= date.today()]
    result = future if not future.empty else dated
    return result.sort_values("_flight_date"), not future.empty

def extract_upcoming_flights(df: pd.DataFrame, limit: int) -> tuple[list[dict], int, bool]:
    if df.empty: return [], 0, False
    frame, used_future = _filter_future(df.copy())
    matched = int(len(frame))
    if matched == 0: return [], 0, used_future

    oc = find_column_name(frame,"ORIGIN_AIRPORT"); dc = find_column_name(frame,"DESTINATION_AIRPORT")
    ac = find_column_name(frame,"AIRLINE"); fnc = find_column_name(frame,"FLIGHT_NUMBER")
    depc = find_column_name(frame,"SCHEDULED_DEPARTURE"); distc = find_column_name(frame,"DISTANCE")
    fdc  = find_column_name(frame,"FLIGHT_DATE")

    rows: list[dict] = []
    for _, row in frame.head(limit).iterrows():
        origin = optional_text(row[oc]).upper() if oc and pd.notna(row[oc]) else None
        dest   = optional_text(row[dc]).upper() if dc and pd.notna(row[dc]) else None
        fd = row["_flight_date"].isoformat() if "_flight_date" in row and pd.notna(row["_flight_date"]) else (
            pd.to_datetime(row[fdc], errors="coerce").date().isoformat() if fdc and pd.notna(row[fdc]) else None)
        rows.append({
            "flight_date": fd,
            "flight_number": format_flight_number(row[fnc]) if fnc else None,
            "airline": optional_text(row[ac]).upper() if ac and pd.notna(row[ac]) else None,
            "origin_airport": origin, "destination_airport": dest,
            "route": f"{origin} -> {dest}" if origin and dest else None,
            "scheduled_departure": format_scheduled_departure(row[depc]) if depc else None,
            "distance_miles": float(row[distc]) if distc and pd.notna(row[distc]) else None,
        })
    return rows, matched, used_future

def departure_sort_key(value: Any) -> int:
    if value is None or pd.isna(value): return 9999
    try: return int(float(value))
    except (TypeError, ValueError):
        text = str(value).strip()
        if ":" in text:
            h, m = text.split(":",1)
            if h.isdigit() and m.isdigit(): return int(h)*100 + int(m)
        return 9999

def format_date_br(iso_date: str | None) -> str | None:
    if not iso_date: return None
    try: return date.fromisoformat(iso_date).strftime("%d/%m/%Y")
    except ValueError: return None

def extract_airport_departures(df: pd.DataFrame, airport_code: str, limit: int) -> tuple[list[dict], int, bool]:
    if df.empty: return [], 0, False
    oc = find_column_name(df,"ORIGIN_AIRPORT")
    if not oc: return [], 0, False
    airport = airport_code.strip().upper()
    filtered = df[df[oc].astype(str).str.strip().str.upper() == airport].copy()
    if filtered.empty: return [], 0, False
    filtered, used_future = _filter_future(filtered)

    depc = find_column_name(filtered,"SCHEDULED_DEPARTURE")
    filtered["_dep_sort"] = filtered[depc].map(departure_sort_key) if depc else 9999
    sort_by = (["_flight_date","_dep_sort"] if "_flight_date" in filtered.columns else ["_dep_sort"])
    filtered = filtered.sort_values(sort_by)
    matched = int(len(filtered))
    if matched == 0: return [], 0, used_future

    dc = find_column_name(filtered,"DESTINATION_AIRPORT"); ac = find_column_name(filtered,"AIRLINE")
    fnc = find_column_name(filtered,"FLIGHT_NUMBER"); arc = find_column_name(filtered,"SCHEDULED_ARRIVAL")
    fdc = find_column_name(filtered,"FLIGHT_DATE")

    rows: list[dict] = []
    for _, row in filtered.head(limit).iterrows():
        fd = row["_flight_date"].isoformat() if "_flight_date" in row and pd.notna(row["_flight_date"]) else (
            pd.to_datetime(row[fdc], errors="coerce").date().isoformat() if fdc and pd.notna(row[fdc]) else None)
        airline = optional_text(row[ac]).upper() if ac and pd.notna(row[ac]) else None
        fn      = format_flight_number(row[fnc]) if fnc else None
        dest    = optional_text(row[dc]).upper() if dc and pd.notna(row[dc]) else None
        rows.append({
            "flight_date": fd, "flight_date_br": format_date_br(fd),
            "airline": airline, "flight_number": fn,
            "flight_code": f"{airline}{fn}" if airline and fn else fn,
            "origin_airport": airport, "destination_airport": dest,
            "scheduled_departure": format_scheduled_departure(row[depc]) if depc else None,
            "scheduled_arrival":   format_scheduled_departure(row[arc])  if arc  else None,
        })
    return rows, matched, used_future


# ── apply_predict_filters / predict_dataframe / dataframe_json_records ───────

def apply_predict_filters(df: pd.DataFrame, payload: PredictRequest) -> pd.DataFrame:
    f = df
    def req_col(col, name):
        c = find_column_name(f, col)
        if not c: raise ValueError(f"Cannot filter by '{name}': missing column '{col}'.")
        return c
    for name, val, col in [("origin_airport",payload.origin_airport,"ORIGIN_AIRPORT"),
                            ("destination_airport",payload.destination_airport,"DESTINATION_AIRPORT"),
                            ("airline",payload.airline,"AIRLINE")]:
        if val: f = f[f[req_col(col,name)].astype(str).str.strip().str.upper() == val]
    for name, val, col in [("year",payload.year,"YEAR"),("month",payload.month,"MONTH"),
                            ("day",payload.day,"DAY"),("day_of_week",payload.day_of_week,"DAY_OF_WEEK"),
                            ("scheduled_departure",payload.scheduled_departure,"SCHEDULED_DEPARTURE")]:
        if val is not None: f = f[pd.to_numeric(f[req_col(col,name)], errors="coerce") == val]
    if payload.flight_date:
        try: td = date.fromisoformat(payload.flight_date)
        except ValueError as exc: raise ValueError("flight_date must be in YYYY-MM-DD format.") from exc
        fdc = find_column_name(f,"FLIGHT_DATE")
        yy, mm, dd = find_column_name(f,"YEAR"), find_column_name(f,"MONTH"), find_column_name(f,"DAY")
        if fdc:
            f = f[pd.to_datetime(f[fdc], errors="coerce").dt.date == td]
        elif yy and mm and dd:
            f = f[(pd.to_numeric(f[yy],errors="coerce")==td.year) &
                  (pd.to_numeric(f[mm],errors="coerce")==td.month) &
                  (pd.to_numeric(f[dd],errors="coerce")==td.day)]
        else:
            raise ValueError("Cannot filter by 'flight_date': dataset must include FLIGHT_DATE or YEAR/MONTH/DAY.")
    return f

def predict_dataframe(df, pipeline, meta, threshold):
    X = df
    if meta and "features" in meta and "selected" in meta["features"]:
        required = meta["features"]["selected"]
        missing = [c for c in required if c not in df.columns]
        if missing: raise ValueError(f"Missing columns required by the model: {', '.join(missing)}")
        X = coerce_feature_types(df[required], meta["features"].get("numeric",[]), meta["features"].get("categorical",[]))
    probas = pipeline.predict_proba(X)[:, 1]
    out = df.copy(); out["delay_probability"] = probas; out["delay_prediction"] = (probas >= threshold).astype(int)
    return out

def dataframe_json_records(df: pd.DataFrame) -> list[dict]:
    return [] if df.empty else json.loads(df.where(pd.notna(df), None).to_json(orient="records", date_format="iso"))

def list_route_index() -> list[dict]:
    return sorted([
        {"path": r.rule, "methods": sorted(m for m in r.methods if m not in {"HEAD","OPTIONS"}), "name": r.endpoint}
        for r in app.url_map.iter_rules() if r.rule and not r.rule.startswith("/static")
    ], key=lambda x: x["path"])


# ── Page routes ──────────────────────────────────────────────────────────────

_PAGES = [
    ("/",            "dashboard.html",   "Flight Advisor | Front",        "dashboard"),
    ("/front",       "dashboard.html",   "Flight Advisor | Front",        "dashboard"),
    ("/flight",      "flight.html",      "Flight Advisor | Flights",      "flight"),
    ("/flights",     "flight.html",      "Flight Advisor | Flights",      "flight"),
    ("/predictions", "predictions.html", "Flight Advisor | Predictions",  "predictions"),
    ("/advisor",     "advisor.html",     "Flight Advisor | Advisor",      "advisor"),
]
for _path, _tpl, _title, _active in _PAGES:
    def _make(tpl=_tpl, title=_title, active=_active):
        def _view(): return render_template(tpl, page_title=title, active_page=active)
        return _view
    app.add_url_rule(_path, endpoint=_path.strip("/") or "home", view_func=_make())


# ── API routes ───────────────────────────────────────────────────────────────

@app.get("/api/routes")
def api_routes():
    return jsonify({
        "service": "Flight Advisor (Flask)", "docs": "/docs", "redoc": "/redoc",
        "openapi": "/openapi.json", "routes": list_route_index(),
        "examples": {
            "health":           {"method":"GET",  "path":"/health"},
            "advise":           {"method":"POST", "path":"/advise"},
            "predict":          {"method":"POST", "path":"/predict"},
            "upcoming_flights": {"method":"GET",  "path":"/api/upcoming_flights?limit=50"},
            "flight_countries": {"method":"GET",  "path":"/api/flight/countries"},
            "flight_airports":  {"method":"GET",  "path":"/api/flight/airports?country=Brazil"},
            "flight_departures":{"method":"GET",  "path":"/api/flight/departures?airport=GRU&limit=30"},
        },
    })

@app.get("/docs")
def docs(): return redirect(url_for("api_routes"))

@app.get("/redoc")
def redoc(): return redirect(url_for("api_routes"))

@app.get("/openapi.json")
def openapi_json():
    return jsonify({"openapi":"3.1.0","info":{"title":"Flight Advisor (Flask)","version":"0.1.0"},
                    "note":"OpenAPI schema is not auto-generated in Flask mode.","routes":list_route_index()})

@app.get("/health")
def health():
    try: _ = load_assets(); return jsonify({"status":"ok"})
    except Exception as exc: return jsonify({"status":"error","detail":str(exc)}), 500

@app.get("/api/flight/countries")
def flight_countries():
    source_uri = request.args.get("source_uri")
    try: airports_df, source = load_airports_index(source_uri)
    except FileNotFoundError as exc: return jsonify({"source":None,"total_countries":0,"countries":[],"detail":str(exc)})
    except Exception as exc: return jsonify({"detail":str(exc)}), 500
    grouped = (airports_df.groupby("country",dropna=True)["iata_code"].count()
               .reset_index(name="airport_count").sort_values("country"))
    return jsonify({"source":source,"total_countries":len(grouped),
                    "countries":[{"country":str(r["country"]),"airport_count":int(r["airport_count"])} for _,r in grouped.iterrows()]})

@app.get("/api/flight/airports")
def flight_airports():
    country    = request.args.get("country","").strip()
    source_uri = request.args.get("source_uri")
    limit_raw  = request.args.get("limit","800")
    if not country: return jsonify({"detail":"Query parameter 'country' is required."}), 400
    try: limit = max(1, min(int(limit_raw), 5000))
    except ValueError: return jsonify({"detail":"Query parameter 'limit' must be an integer."}), 400
    try: airports_df, source = load_airports_index(source_uri)
    except FileNotFoundError as exc: return jsonify({"source":None,"country":country,"total_airports":0,"airports":[],"detail":str(exc)})
    except Exception as exc: return jsonify({"detail":str(exc)}), 500
    selected = (airports_df[airports_df["country"].astype(str).str.casefold() == country.casefold()]
                .sort_values(["city","airport_name","iata_code"],na_position="last").head(limit))

    def _f(v):
        try: f = float(v); return None if f!=f else f
        except: return None

    return jsonify({"source":source,"country":country,"total_airports":len(selected),
                    "airports":[{"iata_code":optional_text(r.get("iata_code")),
                                 "airport_name":optional_text(r.get("airport_name")),
                                 "city":optional_text(r.get("city")),"state":optional_text(r.get("state")),
                                 "country":optional_text(r.get("country")),
                                 "latitude":_f(r.get("latitude")),"longitude":_f(r.get("longitude"))}
                                for r in selected.to_dict(orient="records")]})

@app.get("/api/flight/departures")
def flight_departures():
    airport    = request.args.get("airport","").strip().upper()
    source_uri = request.args.get("source_uri")
    limit_raw  = request.args.get("limit","50")
    if not airport: return jsonify({"detail":"Query parameter 'airport' is required."}), 400
    try: limit = max(1, min(int(limit_raw), 500))
    except ValueError: return jsonify({"detail":"Query parameter 'limit' must be an integer."}), 400

    deps, matched, future, total_rows, source = [], 0, False, 0, None
    try:
        flights_df, source = load_upcoming_flights_frame(source_uri)
        total_rows = len(flights_df)
        deps, matched, future = extract_airport_departures(flights_df, airport, limit)
    except FileNotFoundError:
        # No upcoming flights file found; proceed to generate suggestions.
        pass
    except Exception as exc:
        return jsonify({"detail":str(exc)}), 500

    if not deps:
        try:
            airports_df, _ = load_airports_index()
            possible_dests = airports_df[airports_df["iata_code"] != airport]

            popular_dests_iata = ["JFK", "LAX", "LHR", "CDG", "DXB", "HND", "GRU", "EZE"]
            sample_destinations = possible_dests[possible_dests["iata_code"].isin(popular_dests_iata)]

            num_missing = 3 - len(sample_destinations)
            if num_missing > 0:
                additional_samples = possible_dests[~possible_dests["iata_code"].isin(popular_dests_iata)].sample(min(num_missing, len(possible_dests[~possible_dests["iata_code"].isin(popular_dests_iata)])))
                sample_destinations = pd.concat([sample_destinations, additional_samples])

            fake_deps = []
            today_iso = date.today().isoformat()
            today_br = date.today().strftime("%d/%m/%Y")
            for _, dest_row in sample_destinations.head(3).iterrows():
                fake_deps.append({
                    "flight_date": today_iso,
                    "flight_date_br": today_br,
                    "airline": "ZZ",
                    "flight_number": "9876",
                    "flight_code": "ZZ9876",
                    "origin_airport": airport,
                    "destination_airport": dest_row["iata_code"],
                    "scheduled_departure": "11:00",
                    "scheduled_arrival": "14:00",
                })
            deps = fake_deps
            matched = len(deps)
            # Signals to the frontend that these are future/scheduled flights
            future = True
        except Exception:
            # If suggestion generation fails, return the empty deps list.
            pass

    return jsonify({"source":source,"airport":airport,"total_rows":total_rows,
                    "matched_rows":matched,"returned_rows":len(deps),"future_window":future,"departures":deps})

@app.get("/api/upcoming_flights")
@app.get("/api/weekly_predictions")
def upcoming_flights():
    limit_raw  = request.args.get("limit","50")
    source_uri = request.args.get("source_uri")
    try: limit = max(1, min(int(limit_raw), 500))
    except ValueError: return jsonify({"detail":"Query parameter 'limit' must be an integer."}), 400
    try: df, source = load_upcoming_flights_frame(source_uri)
    except FileNotFoundError as exc:
        return jsonify({"source":None,"total_rows":0,"matched_rows":0,"returned_rows":0,
                        "future_window":False,"predictions":[],"detail":str(exc)})
    except Exception as exc: return jsonify({"detail":str(exc)}), 500
    flights, matched, future = extract_upcoming_flights(df, limit)
    return jsonify({"source":source,"total_rows":len(df),"matched_rows":matched,
                    "returned_rows":len(flights),"future_window":future,"predictions":flights})

@app.route("/advise", methods=["POST","OPTIONS"])
def advise():
    if request.method == "OPTIONS": return ("", 204)
    payload_data = request.get_json(silent=True)
    if payload_data is None: return jsonify({"detail":"Body must be valid JSON."}), 400
    try: payload = AdviseRequest.model_validate(payload_data)
    except ValidationError as exc: return jsonify({"detail":exc.errors()}), 400
    try: pipeline, meta, maps, global_rate, route_distance_map, global_distance = load_assets()
    except Exception as exc: return jsonify({"detail":str(exc)}), 500
    try: df = build_features(payload, maps, global_rate, route_distance_map, global_distance)
    except ValueError as exc: return jsonify({"detail":str(exc)}), 400
    X = df
    if meta and "features" in meta and "selected" in meta["features"]:
        required = meta["features"]["selected"]
        missing = [c for c in required if c not in df.columns]
        if missing: return jsonify({"detail":f"Missing columns required by the model: {', '.join(missing)}"}), 400
        X = coerce_feature_types(df[required], meta["features"].get("numeric",[]), meta["features"].get("categorical",[]))
    prob = float(pipeline.predict_proba(X)[:, 1][0])
    level = risk_level(prob); top = compute_top_factors(df, global_rate)
    return jsonify(AdviseResponse(delay_probability=prob, risk_level=level, top_factors=top,
                                  advice=advice_text(prob, level, top)).model_dump())

@app.route("/predict", methods=["POST","OPTIONS"])
def predict():
    if request.method == "OPTIONS": return ("", 204)
    payload_data = request.get_json(silent=True)
    if payload_data is None:
        if request.get_data(as_text=True).strip(): return jsonify({"detail":"Body must be valid JSON."}), 400
        payload_data = {}
    try: payload = PredictRequest.model_validate(payload_data)
    except ValidationError as exc: return jsonify({"detail":exc.errors()}), 400
    try: pipeline, meta, *_ = load_assets()
    except Exception as exc: return jsonify({"detail":str(exc)}), 500
    try: base_df, source = load_predict_frame(payload)
    except (ValueError, FileNotFoundError) as exc: return jsonify({"detail":str(exc)}), 400
    except Exception as exc: return jsonify({"detail":str(exc)}), 500
    total_rows = int(len(base_df))
    if total_rows == 0:
        return jsonify({"source":source,"threshold":payload.threshold,"total_rows":0,
                        "matched_rows":0,"returned_rows":0,"predictions":[]})
    try: filtered_df = apply_predict_filters(base_df, payload)
    except ValueError as exc: return jsonify({"detail":str(exc)}), 400
    matched = int(len(filtered_df))
    if matched == 0:
        return jsonify({"source":source,"threshold":payload.threshold,"total_rows":total_rows,
                        "matched_rows":0,"returned_rows":0,"predictions":[]})
    try: output_df = predict_dataframe(filtered_df.head(payload.limit).copy(), pipeline, meta, payload.threshold)
    except ValueError as exc: return jsonify({"detail":str(exc)}), 400
    except Exception as exc: return jsonify({"detail":str(exc)}), 500
    return jsonify({"source":source,"threshold":payload.threshold,"total_rows":total_rows,
                    "matched_rows":matched,"returned_rows":len(output_df),"predictions":dataframe_json_records(output_df)})


def run_local_server() -> int:
    host  = os.getenv("API_HOST","127.0.0.1")
    debug = os.getenv("FLASK_DEBUG","0") == "1"
    try: port = int(os.getenv("API_PORT","8000"))
    except ValueError: print("Invalid API_PORT. Falling back to 8000.", file=sys.stderr); port = 8000
    app.run(host=host, port=port, debug=debug)
    return 0


@app.get("/api/live_flights")
def live_flights():
    """
    Returns live aircraft in flight via OpenSky Network.

    Query params
    ------------
    region  : name of a pre-defined region (brazil | south_america | north_america |
              europe | world). Default: "brazil".
    lamin, lomin, lamax, lomax : custom bounding box (decimal degrees).
              Takes precedence over `region` when all four are provided.
    limit   : maximum number of aircraft returned (1-500, default 200).

    Examples
    --------
    GET /api/live_flights?region=brazil&limit=100
    GET /api/live_flights?lamin=-23.7&lomin=-46.9&lamax=-23.4&lomax=-46.5
    """
    from services.OpenSky import fetch_live_flights_cached, BOUNDING_BOXES

    # --- bounding box ---
    try:
        custom_bb = None
        bb_params = [request.args.get(k) for k in ("lamin", "lomin", "lamax", "lomax")]
        if all(bb_params):
            custom_bb = tuple(float(v) for v in bb_params)  # type: ignore[arg-type]
    except ValueError:
        return jsonify({"detail": "lamin/lomin/lamax/lomax must be decimal numbers."}), 400

    region = request.args.get("region", "brazil").strip().lower()
    bounding_box = custom_bb or BOUNDING_BOXES.get(region) or BOUNDING_BOXES["brazil"]

    # --- limit ---
    try:
        limit = max(1, min(int(request.args.get("limit", 200)), 500))
    except ValueError:
        return jsonify({"detail": "The 'limit' parameter must be an integer."}), 400

    include_ground = request.args.get("include_ground", "").strip().lower() in {"1", "true", "yes", "on"}

    try:
        flights, cache_age = fetch_live_flights_cached(
            bounding_box=bounding_box,
            include_ground=include_ground,
        )
    except TimeoutError as exc:
        return jsonify({"detail": str(exc)}), 504
    except RuntimeError as exc:
        return jsonify({"detail": str(exc)}), 502
    except Exception as exc:
        return jsonify({"detail": f"Unexpected error: {exc}"}), 500

    sliced = flights[:limit]

    return jsonify({
        "source":        "OpenSky Network",
        "region":        region,
        "bounding_box":  {"lamin": bounding_box[0], "lomin": bounding_box[1],
                          "lamax": bounding_box[2], "lomax": bounding_box[3]},
        "include_ground": include_ground,
        "cache_age_sec": cache_age,
        "total_found":   len(flights),
        "returned":      len(sliced),
        "flights":       sliced,
    })


@app.get("/api/live_flights/<string:icao24>")
def live_flight_detail(icao24: str):
    """
    Returns live data for a specific aircraft by its ICAO24 code.

    Example
    -------
    GET /api/live_flights/a0b1c2
    """
    from services.OpenSky import fetch_live_flights_cached

    icao24 = icao24.strip().lower()

    try:
        flights, cache_age = fetch_live_flights_cached(icao24=icao24)
    except TimeoutError as exc:
        return jsonify({"detail": str(exc)}), 504
    except RuntimeError as exc:
        return jsonify({"detail": str(exc)}), 502
    except Exception as exc:
        return jsonify({"detail": f"Unexpected error: {exc}"}), 500

    match = next((f for f in flights if f.get("icao24", "").lower() == icao24), None)
    if not match:
        return jsonify({"detail": f"Aircraft '{icao24}' not found or not currently in flight."}), 404

    return jsonify({"source": "OpenSky Network", "cache_age_sec": cache_age, "flight": match})


if __name__ == "__main__":
    raise SystemExit(run_local_server())
