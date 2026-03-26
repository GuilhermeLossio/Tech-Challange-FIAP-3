#!/usr/bin/env python3
"""Flask app for Flight Advisor (frontend + JSON API)."""
from __future__ import annotations

import json, os, re, sys, unicodedata
from collections import defaultdict
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode
from uuid import uuid4

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
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
from api.services.llm_service import (  # noqa: E402
    generate_llm_advice,
    should_use_llm,
)
from api.views import (  # noqa: E402
    register_advisor_views,
    register_flight_views,
    register_page_views,
)

load_env_file()
app = Flask(__name__, template_folder=str(SRC_DIR / "templates"),
            static_folder=str(SRC_DIR / "static"), static_url_path="/static")
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or "dev-flight-advisor-secret"
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

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
        os.environ.setdefault("DASH_ROUTES_PATH", "/")
        from dashboard.app import app as dash_app
    except Exception as exc:
        print(f"Dash not mounted: {exc}", file=sys.stderr)
        return
    flask_app.wsgi_app = DispatcherMiddleware(flask_app.wsgi_app, {"/dashboard": dash_app.server.wsgi_app})
    flask_app._dash_mounted = True

mount_dash_app(app)


# ── Pydantic models ──────────────────────────────────────────────────────────

class AdviseRequest(BaseModel):
    origin_country: str | None = Field(None, description="Origin country")
    origin_airport: str | None = Field(None, description="Origin IATA code")
    destination_country: str | None = Field(None, description="Destination country")
    destination_airport: str | None = Field(None, description="Destination IATA code")
    airline: str | None = Field(None, description="Airline IATA code")
    scheduled_departure: int | None = Field(None, description="Scheduled departure time (HHMM)", ge=0, le=2359)
    month: int | None = Field(None, ge=1, le=12)
    day_of_week: int | None = Field(None, ge=1, le=7)
    day: int | None = Field(None, ge=1, le=31)
    year: int | None = Field(None, ge=2000, le=2100)
    flight_date: str | None = Field(None, description="YYYY-MM-DD")
    distance: float | None = Field(None, description="Route distance in miles")
    question: str | None = Field(None, description="Free-form user question")

    @field_validator("origin_airport", "destination_airport", "airline")
    @classmethod
    def normalize_codes(cls, v: str | None) -> str | None:
        return v.strip().upper() or None if v else None

    @field_validator("origin_country", "destination_country", "question")
    @classmethod
    def normalize_text_fields(cls, v: str | None) -> str | None:
        return v.strip() or None if v else None


class Factor(BaseModel):
    feature: str
    impact: str


class SuggestedFlight(BaseModel):
    flight_date: str | None = None
    flight_date_br: str | None = None
    airline: str | None = None
    flight_number: str | None = None
    flight_code: str | None = None
    origin_airport: str | None = None
    destination_airport: str | None = None
    scheduled_departure: str | None = None
    delay_probability: float | None = None
    delay_prediction: int | None = None
    risk_level: str | None = None


class RouteDropdownValue(BaseModel):
    country: str | None = None
    airport: str | None = None


class RouteDropdownUpdates(BaseModel):
    origin: RouteDropdownValue | None = None
    destination: RouteDropdownValue | None = None


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: str
    mode: str | None = None
    delay_probability: float | None = None
    delay_prediction: int | None = None
    risk_level: str | None = None
    advice_source: str | None = None
    advice_model: str | None = None
    top_factors: List[Factor] = Field(default_factory=list)
    suggested_flights: List[SuggestedFlight] = Field(default_factory=list)
    clarification_prompts: List[str] = Field(default_factory=list)
    route_updates: RouteDropdownUpdates | None = None


class AdviseResponse(BaseModel):
    delay_probability: float | None = None
    delay_prediction: int | None = None
    risk_level: str | None = None
    top_factors: List[Factor]
    advice: str
    mode: str = "route"
    advice_source: str = "heuristic"
    advice_model: str | None = None
    suggested_flights: List[SuggestedFlight] = Field(default_factory=list)
    clarification_prompts: List[str] = Field(default_factory=list)
    session_id: str | None = None
    messages: List[ChatMessage] = Field(default_factory=list)
    route_updates: RouteDropdownUpdates | None = None


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
DISCOVERY_PROMPTS = [
    "I want to travel to a specific country.",
    "What is the best day or time to fly with lower delay risk?",
    "When is the best time to buy my ticket?",
]
ROUTE_CONTEXT_FIELDS = (
    "origin_country",
    "origin_airport",
    "destination_country",
    "destination_airport",
    "airline",
    "scheduled_departure",
    "flight_date",
    "year",
    "month",
    "day",
    "day_of_week",
    "distance",
)
ADVISOR_CHAT_DIR = ROOT_DIR / "data" / "runtime" / "advisor_sessions"
ADVISOR_CHAT_MAX_MESSAGES = max(4, int(os.getenv("ADVISOR_CHAT_MAX_MESSAGES", "16")))
ADVISOR_CHAT_MAX_CONTENT = max(120, int(os.getenv("ADVISOR_CHAT_MAX_CONTENT", "1200")))


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def allow_data_source_override() -> bool:
    raw = (os.getenv("ALLOW_DATA_SOURCE_OVERRIDE") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def validate_source_override(source_uri: str | None, field_name: str = "source_uri") -> str | None:
    text = str(source_uri or "").strip()
    if not text:
        return None
    if not allow_data_source_override():
        raise ValueError(f"'{field_name}' override is disabled in this environment.")
    return text


def trim_chat_text(value: str | None) -> str:
    text = (value or "").strip()
    if len(text) <= ADVISOR_CHAT_MAX_CONTENT:
        return text
    return text[:ADVISOR_CHAT_MAX_CONTENT].rstrip() + "..."


def chat_bootstrap_message() -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=(
            "I'm the Flight Advisor. I can help with destinations, the best day or time to fly, "
            "delay risk, and when to buy your ticket. Feel free to use the route filters as optional context."
        ),
        created_at=now_iso(),
        mode="discovery",
        advice_source="system",
        clarification_prompts=DISCOVERY_PROMPTS,
    )


def advisor_chat_file(session_id: str) -> Path:
    safe_id = "".join(ch for ch in session_id if ch.isalnum() or ch in {"-", "_"})
    return ADVISOR_CHAT_DIR / f"{safe_id}.json"


def get_advisor_session_id() -> str:
    current = session.get("advisor_session_id")
    if isinstance(current, str) and current.strip():
        return current
    current = uuid4().hex
    session["advisor_session_id"] = current
    session.modified = True
    return current


def load_advisor_messages() -> tuple[str, list[ChatMessage]]:
    session_id = get_advisor_session_id()
    path = advisor_chat_file(session_id)
    if not path.exists():
        return session_id, [chat_bootstrap_message()]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return session_id, [chat_bootstrap_message()]
    if not isinstance(raw, list):
        return session_id, [chat_bootstrap_message()]
    messages: list[ChatMessage] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            messages.append(ChatMessage.model_validate(item))
        except ValidationError:
            continue
    return session_id, messages or [chat_bootstrap_message()]


def persist_advisor_messages(session_id: str, messages: list[ChatMessage]) -> list[ChatMessage]:
    ADVISOR_CHAT_DIR.mkdir(parents=True, exist_ok=True)
    trimmed = messages[-ADVISOR_CHAT_MAX_MESSAGES:]
    advisor_chat_file(session_id).write_text(
        json.dumps([item.model_dump() for item in trimmed], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return trimmed


def reset_advisor_messages() -> tuple[str, list[ChatMessage]]:
    session_id = get_advisor_session_id()
    messages = [chat_bootstrap_message()]
    return session_id, persist_advisor_messages(session_id, messages)


def message_snapshot_for_llm(messages: list[ChatMessage], limit: int = 8) -> list[dict[str, str]]:
    snapshot: list[dict[str, str]] = []
    for item in messages[-limit:]:
        snapshot.append({
            "role": item.role,
            "content": trim_chat_text(item.content),
        })
    return snapshot


def user_chat_message(content: str) -> ChatMessage:
    return ChatMessage(
        role="user",
        content=trim_chat_text(content),
        created_at=now_iso(),
    )


def assistant_chat_message(response: AdviseResponse) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=trim_chat_text(response.advice),
        created_at=now_iso(),
        mode=response.mode,
        delay_probability=response.delay_probability,
        delay_prediction=response.delay_prediction,
        risk_level=response.risk_level,
        advice_source=response.advice_source,
        advice_model=response.advice_model,
        top_factors=response.top_factors,
        suggested_flights=response.suggested_flights,
        clarification_prompts=response.clarification_prompts,
        route_updates=response.route_updates,
    )

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
    today = date.today()
    if req.month is None and req.day_of_week is None and req.day is None:
        return today
    if req.month is None and req.day_of_week is not None:
        return today + timedelta(days=(req.day_of_week - (today.weekday() + 1)) % 7)
    month = req.month or today.month
    year = req.year or (today.year + (1 if month < today.month else 0))
    if req.day is not None:
        return date(year, month, req.day)
    if req.day_of_week is None:
        return date(year, month, 1)
    first = date(year, month, 1)
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
    resolved = resolve_prediction_inputs(req, route_distance_map, global_distance)
    route = resolved["route"]
    df = pd.DataFrame([{
        "YEAR": flight_date.year, "MONTH": req.month or flight_date.month,
        "DAY": flight_date.day, "DAY_OF_WEEK": dow,
        "SCHEDULED_DEPARTURE": resolved["scheduled_departure"],
        "DISTANCE": resolved["distance"], "ORIGIN_AIRPORT": req.origin_airport,
        "DESTINATION_AIRPORT": req.destination_airport, "AIRLINE": resolved["airline"],
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


def build_advisor_top_factors(
    df: pd.DataFrame,
    payload: AdviseRequest,
    global_rate: float,
    resolved_inputs: dict[str, Any],
) -> List[Factor]:
    factors: list[Factor] = []
    if not payload.airline:
        factors.append(Factor(
            feature="carrier_fallback",
            impact="No airline was provided, so carrier-specific behavior falls back to route and global historical averages.",
        ))

    distance = resolved_inputs.get("distance")
    distance_source = resolved_inputs.get("distance_source")
    if distance is not None:
        if distance_source == "request":
            impact = f"Used the provided flight distance of {distance:.0f} miles as an input signal."
        elif distance_source == "route_average":
            impact = f"Estimated distance from the historical average for this route: {distance:.0f} miles."
        else:
            impact = f"Route distance was unavailable, so the model used the global average distance of {distance:.0f} miles."
        factors.append(Factor(feature="distance", impact=impact))

    for factor in compute_top_factors(df, global_rate):
        if any(existing.feature == factor.feature for existing in factors):
            continue
        factors.append(factor)
    return factors[:3]

def risk_level(p: float) -> str:
    return "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.4 else "LOW"


def prediction_threshold() -> float:
    try:
        return min(max(float(os.getenv("ADVISOR_DELAY_THRESHOLD", "0.5")), 0.0), 1.0)
    except ValueError:
        return 0.5


def delay_prediction_from_probability(probability: float, threshold: float | None = None) -> int:
    cutoff = prediction_threshold() if threshold is None else threshold
    return int(float(probability) >= cutoff)


def delay_prediction_text(prediction: int | None) -> str:
    if prediction is None:
        return "The model could not classify the delay outcome."
    return (
        "The model predicts this flight is more likely to be delayed."
        if int(prediction) == 1 else
        "The model predicts this flight is more likely to operate on time."
    )


def clamp_scheduled_departure(value: Any, default: int = 1200) -> int:
    try:
        raw = int(float(value))
    except (TypeError, ValueError):
        raw = default
    raw = max(0, min(raw, 2359))
    hour = max(0, min(raw // 100, 23))
    minute = max(0, min(raw % 100, 59))
    return hour * 100 + minute


def default_scheduled_departure() -> int:
    return clamp_scheduled_departure(os.getenv("ADVISOR_DEFAULT_SCHEDULED_DEPARTURE", "1200"))


def default_airline_code() -> str:
    code = (os.getenv("ADVISOR_DEFAULT_AIRLINE") or "UNKNOWN").strip().upper()
    return code or "UNKNOWN"


def resolve_prediction_inputs(
    req: AdviseRequest,
    route_distance_map: Dict[str, float],
    global_distance: float,
) -> dict[str, Any]:
    route = f"{req.origin_airport}_{req.destination_airport}"
    if req.distance is not None:
        distance = float(req.distance)
        distance_source = "request"
    elif route_distance_map.get(route) is not None:
        distance = float(route_distance_map[route])
        distance_source = "route_average"
    else:
        distance = float(global_distance)
        distance_source = "global_average"

    return {
        "route": route,
        "distance": distance,
        "distance_source": distance_source,
        "airline": req.airline or default_airline_code(),
        "airline_source": "request" if req.airline else "fallback",
        "scheduled_departure": (
            clamp_scheduled_departure(req.scheduled_departure)
            if req.scheduled_departure is not None else
            default_scheduled_departure()
        ),
        "scheduled_departure_source": "request" if req.scheduled_departure is not None else "fallback",
    }

def has_any_date_context(req: AdviseRequest) -> bool:
    return bool(req.flight_date or req.year or req.month or req.day or req.day_of_week)

def has_full_route_context(req: AdviseRequest) -> bool:
    return bool(
        req.origin_airport and req.destination_airport and req.airline
        and req.scheduled_departure is not None
    )

def has_specific_flight_prediction_context(req: AdviseRequest) -> bool:
    return has_full_route_context(req) and has_any_date_context(req)

def has_any_route_context(req: AdviseRequest) -> bool:
    return bool(
        req.origin_airport or req.destination_airport or req.airline
        or req.scheduled_departure is not None
    )

def missing_route_fields(req: AdviseRequest) -> list[str]:
    missing: list[str] = []
    if not req.origin_airport:
        missing.append("origin airport")
    if not req.destination_airport:
        missing.append("destination airport")
    return missing


def question_requests_delay_assessment(question: str | None) -> bool:
    text = (question or "").strip().casefold()
    if not text:
        return False

    positive_tokens = (
        "atraso", "delay", "risco", "chance de atraso", "probabilidade",
        "pontual", "pontualidade", "on time", "on-time", "late",
        "melhor dia", "melhor horario", "melhor horário", "menos risco",
        "horario para voar", "horário para voar", "dia para voar",
        "risk", "on-time performance", "best time to fly", "best day to fly",
        "departure time", "flight risk", "schedule risk",
        "previsao", "previsão", "prever", "forecast", "prediction",
        "weekly", "semanal", "semana",
    )
    if any(token in text for token in positive_tokens):
        return True

    negative_tokens = (
        "preco", "precos", "price", "fare", "tarifa", "passagem", "comprar",
        "compra", "destino", "destination", "pais", "country", "viagem",
        "trip", "ferias", "férias", "turismo", "roteiro", "bagagem",
        "documento", "visto", "hotel",
        "ticket", "buy", "purchase", "where to go", "which country",
        "clima", "weather", "temperature", "temperatura", "region", "regiao",
        "região", "activities", "activity", "things to do", "tourist",
        "tourism", "attractions", "atrações", "atracoes",
    )
    if any(token in text for token in negative_tokens):
        return False

    return False


def should_auto_run_delay_prediction(question: str | None) -> bool:
    return not (question or "").strip() or question_requests_delay_assessment(question)


def should_run_route_prediction(req: AdviseRequest) -> bool:
    return has_specific_flight_prediction_context(req) and should_auto_run_delay_prediction(req.question)

def should_run_weekly_route_prediction(req: AdviseRequest) -> bool:
    return bool(
        req.origin_airport and req.destination_airport
        and should_auto_run_delay_prediction(req.question)
        and not should_run_route_prediction(req)
    )

def detect_discovery_topic(question: str | None) -> str:
    text = (question or "").strip().casefold()
    if not text:
        return "general"
    if any(token in text for token in ("preco", "preços", "price", "fare", "tarifa", "passagem")):
        return "price"
    if any(token in text for token in ("pais", "country", "destino", "destination", "viagem", "trip")):
        return "country"
    if any(token in text for token in (
        "dia", "day", "horario", "time", "atraso", "delay", "melhor voo", "best flight",
        "previsao", "previsão", "prever", "forecast", "prediction",
        "weekly", "semanal", "semana",
    )):
        return "timing"
    return "general"

def build_discovery_response(req: AdviseRequest) -> AdviseResponse:
    topic = detect_discovery_topic(req.question)
    missing = missing_route_fields(req) if has_any_route_context(req) else []

    if missing:
        advice = (
            "Before I analyze a specific flight, I still need "
            + ", ".join(missing)
            + ". If you prefer, tell me whether the customer wants a destination country, "
              "a better day or time to fly, or guidance on when to buy the ticket."
        )
    elif topic == "country":
        advice = (
            "Start by selecting the destination country. After that, choose the airport from the dropdown "
            "so I can narrow the route without overloading the list."
        )
    elif topic == "timing":
        advice = (
            "To compare lower-risk options, start with origin and destination airports. "
            "If airline, departure time, or date are missing, I can use the upcoming weekly schedule instead "
            "of blocking the analysis."
        )
    elif topic == "price":
        advice = (
            "I can ask about the best moment to buy, but this project still does not have historical fare data. "
            "For now I can only give general planning guidance, not evidence-based advice from ticket prices."
        )
    else:
        advice = (
            "Before recommending a flight, tell me what the customer is looking for. "
            "You can ask for a trip to a specific country, the best day or time to fly, "
            "or the best moment to buy a ticket."
        )

    return AdviseResponse(
        top_factors=[],
        advice=advice,
        mode="discovery",
        advice_source="discovery",
        clarification_prompts=DISCOVERY_PROMPTS,
    )

def advisor_env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

def advisor_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def advisor_env_text(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip() or default

def build_advisor_runtime_context() -> dict[str, Any]:
    realtime_enabled = advisor_env_flag("ADVISOR_SEARCH_FLIGHTS_ENABLED", "0")
    booking_enabled = advisor_env_flag("ADVISOR_BOOKING_ENABLED", "0")
    fare_history_enabled = advisor_env_flag("ADVISOR_FARE_HISTORY_ENABLED", "0")
    return {
        "role": "Flight Advisor",
        "defaults": {
            "assumed_passengers": {
                "adults": max(1, advisor_env_int("ADVISOR_DEFAULT_ADULTS", 1)),
                "children": max(0, advisor_env_int("ADVISOR_DEFAULT_CHILDREN", 0)),
                "infants": max(0, advisor_env_int("ADVISOR_DEFAULT_INFANTS", 0)),
            },
            "default_cabin_class": advisor_env_text("ADVISOR_DEFAULT_CABIN_CLASS", "economy"),
            "missing_date_strategy": "suggest_nearby_dates_with_good_value",
            "missing_origin_strategy": "ask_naturally_or_use_conversation_context",
            "missing_destination_strategy": "suggest_destinations_from_profile_or_interest",
        },
        "tooling": {
            "search_flights": {
                "enabled": realtime_enabled,
                "name": "search_flights",
                "purpose": "Queries flights available in real time.",
                "parameters": [
                    "origin",
                    "destination",
                    "departure_date",
                    "return_date",
                    "passengers",
                    "cabin_class",
                ],
                "status": (
                    "integrated_realtime_search"
                    if realtime_enabled else
                    "not_integrated_in_this_backend"
                ),
            },
            "booking_flow": {
                "enabled": booking_enabled,
                "status": (
                    "can_guide_purchase_steps"
                    if booking_enabled else
                    "informational_only_no_transaction_execution"
                ),
                "requires_explicit_customer_confirmation": True,
            },
            "fare_history": {
                "enabled": fare_history_enabled,
                "status": (
                    "historical_fare_data_available"
                    if fare_history_enabled else
                    "no_historical_fare_dataset_available"
                ),
            },
        },
        "restrictions": {
            "never_invent_prices_or_live_availability": True,
            "never_confirm_purchase_without_explicit_confirmation": True,
            "do_not_store_sensitive_data_beyond_session": True,
        },
    }

def detect_discovery_topic_runtime(question: str | None) -> str:
    text = (question or "").strip().casefold()
    if not text:
        return "general"
    if any(token in text for token in (
        "preco", "precos", "price", "fare", "tarifa", "passagem",
        "ticket", "buy", "purchase", "cost", "cheap",
    )):
        return "price"
    if any(token in text for token in (
        "pais", "country", "destino", "destination", "viagem", "viajar",
        "trip", "lisboa", "travel", "where", "onde", "city", "cidade",
        "visit", "visitar", "vacation", "holiday",
        "clima", "weather", "temperature", "temperatura", "region", "regiao",
        "região", "activities", "activity", "things to do", "tourist",
        "tourism", "attractions", "atrações", "atracoes",
    )):
        return "country"
    if any(token in text for token in (
        "dia para voar", "day to fly", "best day to fly",
        "horario para voar", "horário para voar", "time to fly",
        "departure time", "scheduled departure", "atraso", "delay",
        "melhor voo", "best flight", "schedule", "on-time", "on time",
        "pontual", "risco", "risk", "less risk", "lower delay risk",
        "previsao", "previsão", "prever", "forecast", "prediction",
        "weekly", "semanal", "semana",
    )):
        return "timing"
    return "general"


def clean_context_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def build_destination_context(req: AdviseRequest) -> dict[str, Any]:
    context: dict[str, Any] = {
        "country": req.destination_country,
        "airport": req.destination_airport,
    }
    if not req.destination_airport:
        return {key: value for key, value in context.items() if value}

    try:
        airports_df, _ = load_airports_index()
    except Exception:
        return {key: value for key, value in context.items() if value}

    matches = airports_df[
        airports_df["iata_code"].astype(str).str.strip().str.upper() == req.destination_airport
    ]
    if matches.empty:
        return {key: value for key, value in context.items() if value}

    row = matches.iloc[0]
    context["city"] = clean_context_text(row.get("city"))
    context["airport_name"] = clean_context_text(row.get("airport_name"))
    context["country"] = context["country"] or clean_context_text(row.get("country"))
    return {key: value for key, value in context.items() if value}


def destination_context_label(context: dict[str, Any]) -> str | None:
    city = context.get("city")
    country = context.get("country")
    airport_name = context.get("airport_name")
    airport = context.get("airport")
    if city and country:
        return f"{city}, {country}"
    return city or country or airport_name or airport

def build_discovery_response_runtime(req: AdviseRequest) -> AdviseResponse:
    topic = detect_discovery_topic_runtime(req.question)
    missing = missing_route_fields(req) if has_any_route_context(req) else []
    destination_context = build_destination_context(req)
    destination_label = destination_context_label(destination_context)

    if missing:
        advice = (
            "Before I can analyze a specific flight, I still need: "
            + ", ".join(missing)
            + ". Once I have origin and destination, I can estimate the route using the upcoming weekly "
              "schedule even when airline, departure time, or exact date are missing."
        )
    elif topic == "country":
        if destination_label:
            advice = (
                f"We can start with {destination_label}. I can help in a broader, friendlier way "
                "with the typical climate, what the region is like, and activities that fit the trip, "
                "before we move into flight-specific details."
            )
        else:
            advice = (
                "We can start with the destination itself. Tell me the country or city you have in mind "
                "and I can help with the typical climate, regional highlights, and activities that fit the trip."
            )
    elif topic == "timing":
        advice = (
            "I can help find the best day or time to fly with lower delay risk. "
            "With origin and destination I can already use the weekly schedule; airline, departure time, "
            "and exact date only help narrow the answer further."
        )
    elif topic == "price":
        advice = (
            "I can give general guidance on the best time to buy, but this project does not yet "
            "have historical fare data or a live search_flights integration for real-time prices."
        )
    elif has_full_route_context(req):
        place_text = destination_label or "the destination"
        advice = (
            f"I already have {place_text} as context. If you want, I can first help in a more exploratory way "
            "with the typical climate, what the region is like, and activity ideas, without jumping straight "
            "into delay data."
        )
    else:
        advice = (
            "Tell me a bit about the trip you have in mind. I can start in a broader, friendlier way "
            "with destination ideas, the typical climate, regional highlights, and tourist activities, "
            "and only get into delay risk if you want that."
        )

    return AdviseResponse(
        top_factors=[],
        advice=advice,
        mode="discovery",
        advice_source="discovery",
        clarification_prompts=DISCOVERY_PROMPTS,
    )

def build_discovery_context(req: AdviseRequest, fallback_advice: str,
                            messages: list[ChatMessage]) -> dict[str, Any]:
    destination_context = build_destination_context(req)
    return {
        "mode": "discovery",
        "question": req.question or "No explicit question was provided.",
        "route_context_status": {
            "is_complete": has_full_route_context(req),
            "prediction_ready": should_run_route_prediction(req),
            "missing_fields": missing_route_fields(req),
        },
        "partial_request": {
            "origin_country": req.origin_country,
            "origin_airport": req.origin_airport,
            "destination_country": req.destination_country,
            "destination_airport": req.destination_airport,
            "airline": req.airline,
            "scheduled_departure": req.scheduled_departure,
            "flight_date": req.flight_date,
            "year": req.year,
            "month": req.month,
            "day": req.day,
            "day_of_week": req.day_of_week,
            "distance_miles": req.distance,
        },
        "destination_context": destination_context,
        "open_question_guidance": {
            "friendly_discovery": True,
            "avoid_delay_data_unless_explicitly_requested": True,
            "allowed_topics": [
                "typical climate",
                "regional profile",
                "tourist activities",
                "best season in general terms",
                "travel style fit",
            ],
            "weather_scope": "general regional guidance only, not live weather",
        },
        "clarification_prompts": DISCOVERY_PROMPTS,
        "assistant_runtime": build_advisor_runtime_context(),
        "fallback_advice": fallback_advice,
    }

def advice_text(prob: float, level: str, top_factors: List[Factor],
                suggested_flights: List[SuggestedFlight] | None = None,
                delay_prediction: int | None = None) -> str:
    prediction = delay_prediction_from_probability(prob) if delay_prediction is None else int(delay_prediction)
    parts = [f"{delay_prediction_text(prediction)} Estimated delay risk: {prob:.0%} ({level})."]
    if top_factors:
        parts.append(f"The strongest signal is {top_factors[0].feature} ({top_factors[0].impact}).")
    if suggested_flights:
        best = suggested_flights[0]
        option = best.flight_code or best.airline or "another scheduled flight"
        when = f" at {best.scheduled_departure}" if best.scheduled_departure else ""
        risk = f" with {best.delay_probability:.0%} predicted delay risk" if best.delay_probability is not None else ""
        parts.append(f"A lower-risk option in the current schedule is {option}{when}{risk}.")
    return " ".join(parts)

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
    source = validate_source_override(input_uri, "input_uri") or resolve_uri("PREDICT_SOURCE", default)
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
    source_override = validate_source_override(source_uri)
    candidates = [source_override] if source_override else []
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


LOCATION_PAIR_PATTERNS = [
    re.compile(r"\b(?P<origin>[A-Za-z]{3})\s*(?:-|->|>)\s*(?P<destination>[A-Za-z]{3})\b", re.IGNORECASE),
    re.compile(r"\bde\s+(?P<origin>.+?)\s+(?:para|pra|pro|to)\s+(?P<destination>.+?)(?=(?:[,.!?;]|$))", re.IGNORECASE),
]
ORIGIN_LOCATION_PATTERNS = [
    re.compile(
        r"\b(?:saindo|partindo|embarque|origem|from|departing)\s+(?:de|do|da|from)\s+(?P<value>.+?)"
        r"(?=(?:\s+(?:para|pra|pro|destino|to)\b|[,.!?;]|$))",
        re.IGNORECASE,
    ),
    re.compile(r"\borigem\s*[:=-]?\s*(?P<value>.+?)(?=(?:\s+(?:para|pra|pro|destino|to)\b|[,.!?;]|$))", re.IGNORECASE),
]
DESTINATION_LOCATION_PATTERNS = [
    re.compile(r"\b(?:destino|para|pra|pro|to)\s+(?P<value>.+?)(?=(?:[,.!?;]|$))", re.IGNORECASE),
    re.compile(r"\b(?:indo|viajando|voando)\s+(?:para|pra|pro|to)\s+(?P<value>.+?)(?=(?:[,.!?;]|$))", re.IGNORECASE),
]
LOCATION_FRAGMENT_SPLIT_RE = re.compile(
    r"\s+(?:com|sem|usando|pela|por|via|em|no dia|na data|on|with|using)\b",
    re.IGNORECASE,
)
LOCATION_EDGE_ARTICLE_RE = re.compile(r"^(?:o|a|os|as|um|uma)\s+", re.IGNORECASE)


def normalize_location_key(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def clean_location_fragment(value: str | None) -> str:
    text = str(value or "").strip(" \t\r\n,.;:!?()[]{}")
    if not text:
        return ""
    text = LOCATION_FRAGMENT_SPLIT_RE.split(text, maxsplit=1)[0]
    text = LOCATION_EDGE_ARTICLE_RE.sub("", text)
    return text.strip(" \t\r\n,.;:!?()[]{}")


def location_tokens(value: str | None) -> tuple[str, ...]:
    key = normalize_location_key(value)
    return tuple(token for token in key.split(" ") if token)


@lru_cache
def load_airports_search_index(source_uri: str | None = None) -> dict[str, Any]:
    airports_df, source = load_airports_index(source_uri)
    records: list[dict[str, Any]] = []
    iata_index: dict[str, dict[str, Any]] = {}
    country_index: dict[str, str] = {}
    city_index: dict[str, tuple[dict[str, Any], ...]] = {}
    airport_name_index: dict[str, tuple[dict[str, Any], ...]] = {}
    city_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    airport_name_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in airports_df.to_dict(orient="records"):
        iata_code = optional_text(row.get("iata_code"))
        country = optional_text(row.get("country"))
        if not iata_code or not country:
            continue

        city = optional_text(row.get("city"))
        airport_name = optional_text(row.get("airport_name"))
        record = {
            "iata_code": iata_code,
            "country": country,
            "city": city,
            "airport_name": airport_name,
            "country_norm": normalize_location_key(country),
            "city_norm": normalize_location_key(city),
            "airport_name_norm": normalize_location_key(airport_name),
            "city_tokens": frozenset(location_tokens(city)),
            "airport_name_tokens": frozenset(location_tokens(airport_name)),
        }
        records.append(record)
        iata_index[iata_code] = record
        if record["country_norm"]:
            country_index.setdefault(record["country_norm"], country)
        if record["city_norm"]:
            city_groups[record["city_norm"]].append(record)
        if record["airport_name_norm"]:
            airport_name_groups[record["airport_name_norm"]].append(record)

    for key, grouped_records in city_groups.items():
        city_index[key] = tuple(grouped_records)
    for key, grouped_records in airport_name_groups.items():
        airport_name_index[key] = tuple(grouped_records)

    return {
        "source": source,
        "records": tuple(records),
        "iata_index": iata_index,
        "country_index": country_index,
        "city_index": city_index,
        "airport_name_index": airport_name_index,
    }


def _dedupe_location_records(records: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        code = str(record.get("iata_code") or "").strip().upper()
        if code and code not in unique:
            unique[code] = record
    return list(unique.values())


def _build_dropdown_value(records: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> RouteDropdownValue | None:
    unique = _dedupe_location_records(records)
    if not unique:
        return None
    if len(unique) == 1:
        record = unique[0]
        return RouteDropdownValue(country=record.get("country"), airport=record.get("iata_code"))
    countries = {record.get("country") for record in unique if record.get("country")}
    if len(countries) == 1:
        return RouteDropdownValue(country=next(iter(countries)), airport=None)
    return None


def _location_token_match(fragment_tokens: frozenset[str], record_tokens: frozenset[str]) -> bool:
    if not fragment_tokens or not record_tokens:
        return False
    if len(fragment_tokens) == 1:
        token = next(iter(fragment_tokens))
        if len(token) < 4:
            return False
    return fragment_tokens.issubset(record_tokens) or record_tokens.issubset(fragment_tokens)


def _best_partial_location_records(
    fragment_key: str,
    fragment_tokens: frozenset[str],
    records: tuple[dict[str, Any], ...],
    key_name: str,
    token_name: str,
) -> list[dict[str, Any]]:
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for record in records:
        record_key = str(record.get(key_name) or "")
        record_tokens = record.get(token_name) or frozenset()
        if not record_key:
            continue
        if fragment_key in record_key:
            scored.append((len(record_tokens), len(record_key), record))
            continue
        if len(fragment_tokens) > 1 and record_key in fragment_key:
            scored.append((len(record_tokens), len(record_key), record))
            continue
        if _location_token_match(fragment_tokens, record_tokens):
            scored.append((len(record_tokens), len(record_key), record))

    if not scored:
        return []

    best_token_count, best_key_length, _ = max(scored, key=lambda item: (item[0], item[1]))
    return [
        record
        for token_count, key_length, record in scored
        if token_count == best_token_count and key_length == best_key_length
    ]


def resolve_location_fragment(fragment: str | None, source_uri: str | None = None) -> RouteDropdownValue | None:
    cleaned = clean_location_fragment(fragment)
    if not cleaned:
        return None

    try:
        search_index = load_airports_search_index(source_uri)
    except Exception:
        return None
    iata_codes = re.findall(r"\b([A-Za-z0-9]{3})\b", cleaned.upper())
    for iata_code in iata_codes:
        record = search_index["iata_index"].get(iata_code)
        if record:
            return RouteDropdownValue(country=record.get("country"), airport=record.get("iata_code"))

    fragment_key = normalize_location_key(cleaned)
    if not fragment_key:
        return None

    airport_exact = _build_dropdown_value(search_index["airport_name_index"].get(fragment_key, ()))
    if airport_exact:
        return airport_exact

    city_exact = _build_dropdown_value(search_index["city_index"].get(fragment_key, ()))
    if city_exact:
        return city_exact

    country = search_index["country_index"].get(fragment_key)
    if country:
        return RouteDropdownValue(country=country, airport=None)

    fragment_tokens = frozenset(fragment_key.split())
    partial_airports = _best_partial_location_records(
        fragment_key,
        fragment_tokens,
        search_index["records"],
        "airport_name_norm",
        "airport_name_tokens",
    )
    airport_partial = _build_dropdown_value(partial_airports)
    if airport_partial:
        return airport_partial

    partial_cities = _best_partial_location_records(
        fragment_key,
        fragment_tokens,
        search_index["records"],
        "city_norm",
        "city_tokens",
    )
    return _build_dropdown_value(partial_cities)


def extract_route_updates_from_question(question: str | None, source_uri: str | None = None) -> RouteDropdownUpdates | None:
    text = str(question or "").strip()
    if not text:
        return None

    origin: RouteDropdownValue | None = None
    destination: RouteDropdownValue | None = None

    for pattern in LOCATION_PAIR_PATTERNS:
        for match in pattern.finditer(text):
            pair_origin = resolve_location_fragment(match.groupdict().get("origin"), source_uri)
            pair_destination = resolve_location_fragment(match.groupdict().get("destination"), source_uri)
            if pair_origin and pair_destination:
                origin = pair_origin
                destination = pair_destination
                break
        if origin or destination:
            break

    if origin is None:
        for pattern in ORIGIN_LOCATION_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            origin = resolve_location_fragment(match.groupdict().get("value"), source_uri)
            if origin:
                break

    if destination is None:
        for pattern in DESTINATION_LOCATION_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            destination = resolve_location_fragment(match.groupdict().get("value"), source_uri)
            if destination:
                break

    if not origin and not destination:
        return None
    return RouteDropdownUpdates(origin=origin, destination=destination)


def enrich_route_payload_from_question(req: AdviseRequest, source_uri: str | None = None) -> AdviseRequest:
    updates = extract_route_updates_from_question(req.question, source_uri)
    if not updates:
        return req

    # When the user mentions a new route in free text, treat it as the source
    # of truth and discard stale route filters carried in the form/session.
    changed_fields: dict[str, Any] = {field: None for field in ROUTE_CONTEXT_FIELDS}
    if updates.origin:
        changed_fields["origin_country"] = updates.origin.country
        changed_fields["origin_airport"] = updates.origin.airport
    if updates.destination:
        changed_fields["destination_country"] = updates.destination.country
        changed_fields["destination_airport"] = updates.destination.airport
    return req.model_copy(update=changed_fields)


def airport_country_from_code(airport_code: str | None, source_uri: str | None = None) -> str | None:
    code = optional_text(airport_code)
    if not code:
        return None
    try:
        search_index = load_airports_search_index(source_uri)
    except Exception:
        return None
    record = search_index["iata_index"].get(code.strip().upper())
    return optional_text(record.get("country")) if record else None


def route_updates_from_request(req: AdviseRequest) -> RouteDropdownUpdates | None:
    origin = None
    destination = None
    if req.origin_country or req.origin_airport:
        origin = RouteDropdownValue(
            country=req.origin_country or airport_country_from_code(req.origin_airport),
            airport=req.origin_airport,
        )
    if req.destination_country or req.destination_airport:
        destination = RouteDropdownValue(
            country=req.destination_country or airport_country_from_code(req.destination_airport),
            airport=req.destination_airport,
        )
    if not origin and not destination:
        return None
    return RouteDropdownUpdates(origin=origin, destination=destination)


# ── Upcoming flights helpers ─────────────────────────────────────────────────

def resolve_upcoming_flights_sources(source_uri: str | None = None) -> list[str]:
    bucket = os.getenv("S3_BUCKET") or os.getenv("S3_Bucket")
    rp = os.getenv("S3_REFINED_PREFIX","refined")
    source_override = validate_source_override(source_uri)
    candidates = [source_override] if source_override else []
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


def build_suggested_flights(payload: AdviseRequest, pipeline: Any, meta: dict | None,
                            limit: int = 3) -> list[SuggestedFlight]:
    if os.getenv("ADVISOR_SUGGESTIONS_ENABLED", "1") != "1":
        return []
    try:
        flights_df, _ = load_upcoming_flights_frame()
    except Exception:
        return []
    if flights_df.empty:
        return []

    attempts = [
        PredictRequest(
            origin_airport=payload.origin_airport,
            destination_airport=payload.destination_airport,
            flight_date=payload.flight_date,
            year=payload.year,
            month=payload.month,
            day=payload.day,
            day_of_week=payload.day_of_week,
            limit=max(limit * 20, 50),
        ),
        PredictRequest(
            origin_airport=payload.origin_airport,
            destination_airport=payload.destination_airport,
            limit=max(limit * 20, 50),
        ),
    ]

    dep_col = find_column_name(flights_df, "SCHEDULED_DEPARTURE")
    airline_col = find_column_name(flights_df, "AIRLINE")
    fn_col = find_column_name(flights_df, "FLIGHT_NUMBER")
    suggestions: list[SuggestedFlight] = []
    seen: set[tuple[str | None, str | None, str | None, str | None]] = set()

    for filters in attempts:
        try:
            filtered = apply_predict_filters(flights_df, filters)
        except ValueError:
            continue
        if filtered.empty:
            continue

        filtered = filtered.copy()
        if airline_col and dep_col and payload.airline and payload.scheduled_departure is not None:
            same_airline = filtered[airline_col].astype(str).str.strip().str.upper() == payload.airline
            same_departure = pd.to_numeric(filtered[dep_col], errors="coerce") == int(payload.scheduled_departure)
            filtered = filtered[~(same_airline & same_departure)]
        if filtered.empty:
            continue

        try:
            ranked = predict_dataframe(filtered.copy(), pipeline, meta, threshold=0.5)
        except Exception:
            return []

        parsed_dates, ranked = _parse_dates_col(ranked)
        ranked = ranked.copy()
        if parsed_dates is not None:
            ranked["_flight_date"] = parsed_dates
        ranked["_dep_sort"] = ranked[dep_col].map(departure_sort_key) if dep_col else 9999
        ranked["_same_airline"] = (
            ranked[airline_col].astype(str).str.strip().str.upper() == payload.airline
            if airline_col else False
        )

        sort_cols = ["delay_probability", "_same_airline", "_dep_sort"]
        ascending = [True, False, True]
        if "_flight_date" in ranked.columns:
            sort_cols.insert(1, "_flight_date")
            ascending.insert(1, True)
        ranked = ranked.sort_values(sort_cols, ascending=ascending, na_position="last")

        for _, row in ranked.iterrows():
            flight_date_iso = None
            if "_flight_date" in row and pd.notna(row["_flight_date"]):
                flight_date_iso = row["_flight_date"].isoformat()
            airline = optional_text(row[airline_col]).upper() if airline_col and pd.notna(row[airline_col]) else None
            flight_number = format_flight_number(row[fn_col]) if fn_col and pd.notna(row[fn_col]) else None
            departure = format_scheduled_departure(row[dep_col]) if dep_col and pd.notna(row[dep_col]) else None
            dedupe_key = (flight_date_iso, airline, flight_number, departure)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            prob = float(row["delay_probability"]) if pd.notna(row.get("delay_probability")) else None
            suggestions.append(SuggestedFlight(
                flight_date=flight_date_iso,
                flight_date_br=format_date_br(flight_date_iso),
                airline=airline,
                flight_number=flight_number,
                flight_code=f"{airline}{flight_number}" if airline and flight_number else (flight_number or airline),
                origin_airport=payload.origin_airport,
                destination_airport=payload.destination_airport,
                scheduled_departure=departure,
                delay_probability=prob,
                delay_prediction=delay_prediction_from_probability(prob) if prob is not None else None,
                risk_level=risk_level(prob) if prob is not None else None,
            ))
            if len(suggestions) >= limit:
                return suggestions

    return suggestions


def compute_model_delay_snapshot(
    payload: AdviseRequest,
    pipeline: Any,
    meta: dict | None,
    maps: Dict[str, Dict[str, float]],
    global_rate: float,
    route_distance_map: Dict[str, float],
    global_distance: float,
    limit: int = 3,
) -> dict[str, Any]:
    resolved_inputs = resolve_prediction_inputs(payload, route_distance_map, global_distance)
    df = build_features(payload, maps, global_rate, route_distance_map, global_distance)

    X = df
    if meta and "features" in meta and "selected" in meta["features"]:
        required = meta["features"]["selected"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns required by the model: {', '.join(missing)}")
        X = coerce_feature_types(
            df[required],
            meta["features"].get("numeric", []),
            meta["features"].get("categorical", []),
        )

    probability = float(pipeline.predict_proba(X)[:, 1][0])
    prediction = delay_prediction_from_probability(probability)
    level = risk_level(probability)
    top_factors = build_advisor_top_factors(df, payload, global_rate, resolved_inputs)
    suggested_flights = build_suggested_flights(payload, pipeline, meta, limit=limit)

    return {
        "dataframe": df,
        "delay_probability": probability,
        "delay_prediction": prediction,
        "risk_level": level,
        "top_factors": top_factors,
        "suggested_flights": suggested_flights,
        "resolved_inputs": resolved_inputs,
    }


def weekly_prediction_window_days() -> int:
    try:
        return max(1, min(int(os.getenv("ADVISOR_WEEKLY_WINDOW_DAYS", "7")), 14))
    except ValueError:
        return 7


def build_weekly_route_factors(
    ranked: pd.DataFrame,
    payload: AdviseRequest,
    window_days: int,
    airline_col: str | None,
    dep_col: str | None,
) -> list[Factor]:
    factors = [
        Factor(
            feature="schedule_window",
            impact=f"Analyzed {len(ranked)} scheduled flight(s) in the next {window_days} day(s).",
        )
    ]

    if "_flight_date" in ranked.columns and not has_any_date_context(payload):
        by_day = (
            ranked.dropna(subset=["_flight_date"])
            .groupby("_flight_date")["delay_probability"]
            .mean()
            .sort_values()
        )
        if not by_day.empty:
            best_day = by_day.index[0]
            factors.append(Factor(
                feature="lowest_risk_day",
                impact=f"{best_day.strftime('%d/%m/%Y')} has the lowest average predicted risk in the weekly window.",
            ))

    if airline_col and not payload.airline:
        airline_ranking = (
            ranked.dropna(subset=[airline_col])
            .groupby(airline_col)["delay_probability"]
            .mean()
            .sort_values()
        )
        if not airline_ranking.empty:
            best_airline = str(airline_ranking.index[0]).strip().upper()
            factors.append(Factor(
                feature="lowest_risk_airline",
                impact=f"{best_airline} has the lowest average predicted delay risk among the available airlines.",
            ))

    if dep_col and payload.scheduled_departure is None:
        best_departure_row = ranked.dropna(subset=[dep_col]).sort_values("delay_probability").head(1)
        if not best_departure_row.empty:
            departure = format_scheduled_departure(best_departure_row.iloc[0][dep_col])
            if departure:
                factors.append(Factor(
                    feature="lower_risk_departure",
                    impact=f"The current weekly schedule shows lower-risk departures around {departure}.",
                ))

    return factors[:3]


def build_weekly_route_prediction(
    payload: AdviseRequest,
    pipeline: Any,
    meta: dict | None,
    limit: int = 3,
) -> AdviseResponse | None:
    try:
        flights_df, _ = load_upcoming_flights_frame()
    except Exception:
        return None
    if flights_df.empty:
        return None

    filters = PredictRequest(
        origin_airport=payload.origin_airport,
        destination_airport=payload.destination_airport,
        airline=payload.airline,
        flight_date=payload.flight_date,
        year=payload.year,
        month=payload.month,
        day=payload.day,
        day_of_week=payload.day_of_week,
        scheduled_departure=payload.scheduled_departure,
        limit=max(limit * 40, 120),
    )

    try:
        filtered = apply_predict_filters(flights_df, filters)
    except ValueError:
        return None
    if filtered.empty:
        return None

    filtered = filtered.copy()
    filtered, _ = _filter_future(filtered)
    if filtered.empty:
        return None

    window_days = weekly_prediction_window_days()
    if "_flight_date" in filtered.columns and not has_any_date_context(payload):
        window_end = date.today() + timedelta(days=window_days - 1)
        weekly_slice = filtered[filtered["_flight_date"] <= window_end].copy()
        if not weekly_slice.empty:
            filtered = weekly_slice
    if filtered.empty:
        return None

    try:
        ranked = predict_dataframe(filtered.copy(), pipeline, meta, threshold=0.5)
    except Exception:
        return None
    if ranked.empty or "delay_probability" not in ranked.columns:
        return None

    dep_col = find_column_name(ranked, "SCHEDULED_DEPARTURE")
    airline_col = find_column_name(ranked, "AIRLINE")
    fn_col = find_column_name(ranked, "FLIGHT_NUMBER")
    ranked = ranked.copy()
    ranked["_dep_sort"] = ranked[dep_col].map(departure_sort_key) if dep_col else 9999
    sort_cols = ["delay_probability", "_dep_sort"]
    ascending = [True, True]
    if "_flight_date" in ranked.columns:
        sort_cols.insert(1, "_flight_date")
        ascending.insert(1, True)
    ranked = ranked.sort_values(sort_cols, ascending=ascending, na_position="last")

    probability = float(ranked["delay_probability"].mean())
    prediction = delay_prediction_from_probability(probability)
    level = risk_level(probability)
    top_factors = build_weekly_route_factors(ranked, payload, window_days, airline_col, dep_col)

    suggestions: list[SuggestedFlight] = []
    seen: set[tuple[str | None, str | None, str | None, str | None]] = set()
    for _, row in ranked.iterrows():
        flight_date_iso = None
        if "_flight_date" in row and pd.notna(row["_flight_date"]):
            flight_date_iso = row["_flight_date"].isoformat()
        airline = optional_text(row[airline_col]).upper() if airline_col and pd.notna(row[airline_col]) else None
        flight_number = format_flight_number(row[fn_col]) if fn_col and pd.notna(row[fn_col]) else None
        departure = format_scheduled_departure(row[dep_col]) if dep_col and pd.notna(row[dep_col]) else None
        dedupe_key = (flight_date_iso, airline, flight_number, departure)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        suggestions.append(SuggestedFlight(
            flight_date=flight_date_iso,
            flight_date_br=format_date_br(flight_date_iso),
            airline=airline,
            flight_number=flight_number,
            flight_code=f"{airline}{flight_number}" if airline and flight_number else (flight_number or airline),
            origin_airport=payload.origin_airport,
            destination_airport=payload.destination_airport,
            scheduled_departure=departure,
            delay_probability=float(row["delay_probability"]) if pd.notna(row.get("delay_probability")) else None,
            delay_prediction=(
                delay_prediction_from_probability(float(row["delay_probability"]))
                if pd.notna(row.get("delay_probability")) else None
            ),
            risk_level=risk_level(float(row["delay_probability"])) if pd.notna(row.get("delay_probability")) else None,
        ))
        if len(suggestions) >= limit:
            break

    route_label = f"{payload.origin_airport} -> {payload.destination_airport}"
    sample_size = int(len(ranked))
    advice_parts = [
        f"{delay_prediction_text(prediction)} For {route_label}, the average predicted delay risk across {sample_size} scheduled flight(s) in the next {window_days} day(s) is {probability:.0%} ({level})."
    ]
    if suggestions:
        best = suggestions[0]
        when = " | ".join(part for part in [best.flight_date_br or best.flight_date, best.scheduled_departure] if part)
        option = best.flight_code or best.airline or "the current lowest-risk option"
        if when:
            advice_parts.append(f"The lowest-risk option in the weekly schedule is {option} on {when}.")
        else:
            advice_parts.append(f"The lowest-risk option in the weekly schedule is {option}.")
    if not payload.airline:
        advice_parts.append("Because no airline was specified, this estimate uses all airlines available for the route.")
    if not has_any_date_context(payload):
        advice_parts.append("Because no exact date was specified, the estimate uses the upcoming weekly schedule.")

    return AdviseResponse(
        delay_probability=probability,
        delay_prediction=prediction,
        risk_level=level,
        top_factors=top_factors,
        advice=" ".join(advice_parts),
        mode="weekly_route",
        advice_source="weekly_model",
        suggested_flights=suggestions,
    )


def build_advice_context(payload: AdviseRequest, probability: float, level: str,
                         delay_prediction: int | None,
                         top_factors: List[Factor], fallback_advice: str,
                         suggested_flights: List[SuggestedFlight],
                         messages: list[ChatMessage]) -> dict[str, Any]:
    destination_context = build_destination_context(payload)
    return {
        "question": payload.question or "No explicit question was provided.",
        "route_context_status": {
            "is_complete": has_full_route_context(payload),
            "prediction_ready": should_run_route_prediction(payload),
            "missing_fields": missing_route_fields(payload),
        },
        "requested_flight": {
            "origin_country": payload.origin_country,
            "origin_airport": payload.origin_airport,
            "destination_country": payload.destination_country,
            "destination_airport": payload.destination_airport,
            "airline": payload.airline,
            "scheduled_departure": payload.scheduled_departure,
            "flight_date": payload.flight_date,
            "year": payload.year,
            "month": payload.month,
            "day": payload.day,
            "day_of_week": payload.day_of_week,
            "distance_miles": payload.distance,
        },
        "delay_assessment": {
            "delay_probability": round(probability, 4),
            "delay_prediction": delay_prediction,
            "risk_level": level,
            "top_factors": [factor.model_dump() for factor in top_factors],
        },
        "suggested_flights": [flight.model_dump(exclude_none=True) for flight in suggested_flights],
        "destination_context": destination_context,
        "assistant_runtime": build_advisor_runtime_context(),
        "fallback_advice": fallback_advice,
    }


def list_route_index() -> list[dict]:
    return sorted([
        {"path": r.rule, "methods": sorted(m for m in r.methods if m not in {"HEAD","OPTIONS"}), "name": r.endpoint}
        for r in app.url_map.iter_rules() if r.rule and not r.rule.startswith("/static")
    ], key=lambda x: x["path"])


def build_openapi_components() -> dict[str, Any]:
    components: dict[str, Any] = {
        "schemas": {
            "ErrorResponse": {
                "type": "object",
                "properties": {
                    "detail": {"type": "string"},
                },
            },
        },
    }
    for model in (Factor, SuggestedFlight, ChatMessage, AdviseRequest, AdviseResponse, PredictRequest):
        schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
        definitions = schema.pop("$defs", {})
        for name, value in definitions.items():
            components["schemas"].setdefault(name, value)
        components["schemas"][model.__name__] = schema
    return components


def build_api_docs_catalog() -> list[dict[str, Any]]:
    return [
        {
            "tag": "Meta",
            "method": "GET",
            "path": "/health",
            "summary": "Health check",
            "description": "Checks whether the Flask API is up and tries to load the main model assets.",
            "parameters": [],
            "response_schema": {"type": "object"},
            "response_example": {"status": "ok"},
            "error_responses": [
                {"status": 500, "description": "Failed to load assets.", "example": {"status": "error", "detail": "Model assets not found."}},
            ],
        },
        {
            "tag": "Meta",
            "method": "GET",
            "path": "/api/routes",
            "summary": "Registered route index",
            "description": "Lists the registered Flask routes with methods, documentation shortcuts, and quick usage examples.",
            "parameters": [],
            "response_schema": {"type": "object"},
            "response_example": {
                "service": "Flight Advisor (Flask)",
                "docs": "/docs/",
                "redoc": "/redoc/",
                "openapi": "/openapi.json",
                "routes": [{"path": "/health", "methods": ["GET"], "name": "health"}],
            },
        },
        {
            "tag": "Meta",
            "method": "GET",
            "path": "/openapi.json",
            "summary": "OpenAPI schema",
            "description": "Returns the API OpenAPI 3.1 schema for import into tools such as Postman or Insomnia.",
            "parameters": [],
            "response_schema": {"type": "object"},
            "response_example": {"openapi": "3.1.0", "info": {"title": "Flight Advisor (Flask)", "version": "0.1.0"}},
        },
        {
            "tag": "Advisor",
            "method": "GET",
            "path": "/api/advisor/history",
            "summary": "Advisor chat history",
            "description": "Returns the current advisor chat session, including the bootstrap message and previous replies.",
            "parameters": [],
            "response_schema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "messages": {"type": "array", "items": {"$ref": "#/components/schemas/ChatMessage"}},
                },
            },
            "response_example": {
                "session_id": "3b9d22b9a4974c31ad81863fa622f7d3",
                "messages": [{
                    "role": "assistant",
                    "content": "I am Flight Advisor. I can talk about destinations, the best day or time to fly, delay risk, and when it makes sense to buy the ticket.",
                    "created_at": "2026-03-25T11:45:00",
                    "mode": "discovery",
                    "advice_source": "system",
                    "clarification_prompts": [
                        "I want to search for a trip to a specific country.",
                        "What is the best day or time to fly with lower delay risk?",
                        "When is the best time to book the ticket?",
                    ],
                }],
            },
        },
        {
            "tag": "Advisor",
            "method": "POST",
            "path": "/api/advisor/reset",
            "summary": "Reset advisor chat",
            "description": "Clears the history saved in the current session and recreates the conversation from the advisor's initial message.",
            "parameters": [],
            "response_schema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "messages": {"type": "array", "items": {"$ref": "#/components/schemas/ChatMessage"}},
                },
            },
            "response_example": {
                "session_id": "3b9d22b9a4974c31ad81863fa622f7d3",
                "messages": [{
                    "role": "assistant",
                    "content": "I am Flight Advisor. I can talk about destinations, the best day or time to fly, delay risk, and when it makes sense to buy the ticket.",
                    "created_at": "2026-03-25T11:45:00",
                    "mode": "discovery",
                    "advice_source": "system",
                    "clarification_prompts": [
                        "I want to search for a trip to a specific country.",
                        "What is the best day or time to fly with lower delay risk?",
                        "When is the best time to book the ticket?",
                    ],
                }],
            },
        },
        {
            "tag": "Advisor",
            "method": "POST",
            "path": "/advise",
            "summary": "Advisor chat completion",
            "description": "Accepts optional route context and a natural-language question. It keeps messages in the session and can answer in discovery or route mode.",
            "parameters": [],
            "request_schema": {"$ref": "#/components/schemas/AdviseRequest"},
            "request_example": {
                "question": "I want to travel to Lisbon. What is the best day to fly?",
                "origin_country": "Brazil",
                "origin_airport": "GRU",
                "destination_country": "Portugal",
                "destination_airport": "LIS",
                "airline": "TP",
                "scheduled_departure": 2200,
            },
            "response_schema": {"$ref": "#/components/schemas/AdviseResponse"},
            "response_example": {
                "delay_probability": 0.18,
                "risk_level": "LOW",
                "top_factors": [{"feature": "ROUTE_DELAY_RATE", "impact": "Route delay history is below average."}],
                "advice": "For this route, the current risk is low. If you want flexibility, I can compare other nearby departure times.",
                "mode": "route",
                "advice_source": "heuristic",
                "suggested_flights": [{
                    "flight_date": "2026-04-10",
                    "flight_date_br": "10/04/2026",
                    "airline": "TP",
                    "flight_number": "88",
                    "flight_code": "TP88",
                    "origin_airport": "GRU",
                    "destination_airport": "LIS",
                    "scheduled_departure": "22:00",
                    "delay_probability": 0.18,
                    "risk_level": "LOW",
                }],
                "clarification_prompts": [],
                "session_id": "3b9d22b9a4974c31ad81863fa622f7d3",
                "messages": [{
                    "role": "user",
                    "content": "I want to travel to Lisbon. What is the best day to fly?",
                    "created_at": "2026-03-25T11:46:00",
                }],
            },
            "error_responses": [
                {"status": 400, "description": "Invalid JSON body or payload outside the schema.", "example": {"detail": "Body must be valid JSON."}},
                {"status": 500, "description": "Internal failure while loading assets or generating the response.", "example": {"detail": "Unexpected error."}},
            ],
        },
        {
            "tag": "Prediction",
            "method": "POST",
            "path": "/predict",
            "summary": "Delay prediction API",
            "description": "Runs delay prediction for one row, multiple rows, or a CSV/parquet file. It also accepts filters over an input dataset.",
            "parameters": [],
            "request_schema": {"$ref": "#/components/schemas/PredictRequest"},
            "request_example": {
                "origin_airport": "GRU",
                "destination_airport": "JFK",
                "airline": "DL",
                "flight_date": "2026-04-08",
                "scheduled_departure": 2230,
                "limit": 10,
                "threshold": 0.5,
            },
            "response_schema": {"type": "object"},
            "response_example": {
                "source": "file",
                "threshold": 0.5,
                "total_rows": 1384,
                "matched_rows": 12,
                "returned_rows": 10,
                "predictions": [{
                    "ORIGIN_AIRPORT": "GRU",
                    "DESTINATION_AIRPORT": "JFK",
                    "AIRLINE": "DL",
                    "delay_probability": 0.37,
                    "delay_prediction": 0,
                }],
            },
            "error_responses": [
                {"status": 400, "description": "Invalid filter, invalid JSON, or missing dataset.", "example": {"detail": "flight_date must be in YYYY-MM-DD format."}},
                {"status": 500, "description": "Failed to load the pipeline or execute the prediction.", "example": {"detail": "Model assets not found."}},
            ],
        },
        {
            "tag": "Flights",
            "method": "GET",
            "path": "/api/upcoming_flights",
            "summary": "Upcoming predicted flights",
            "description": "Returns future flights or, if none are available, the best available window for the prediction table.",
            "parameters": [
                {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500}, "description": "Maximum number of returned records.", "example": 50},
                {"name": "source_uri", "in": "query", "required": False, "schema": {"type": "string"}, "description": "Administrative source override for the future-flights dataset. Disabled by default unless ALLOW_DATA_SOURCE_OVERRIDE=1.", "example": "data/refined/future_flights.parquet"},
            ],
            "response_schema": {"type": "object"},
            "response_example": {
                "source": "file",
                "total_rows": 1384,
                "matched_rows": 214,
                "returned_rows": 50,
                "future_window": True,
                "predictions": [{
                    "flight_date": "2026-04-03",
                    "flight_number": "88",
                    "airline": "TP",
                    "origin_airport": "GRU",
                    "destination_airport": "LIS",
                    "route": "GRU -> LIS",
                    "scheduled_departure": "22:00",
                    "distance_miles": 4923.0,
                }],
            },
        },
        {
            "tag": "Flights",
            "method": "GET",
            "path": "/api/weekly_predictions",
            "summary": "Legacy alias for upcoming flights",
            "description": "Legacy alias of /api/upcoming_flights. Kept for compatibility with the older frontend.",
            "parameters": [
                {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500}, "description": "Maximum number of returned records.", "example": 50},
                {"name": "source_uri", "in": "query", "required": False, "schema": {"type": "string"}, "description": "Administrative source override for the future-flights dataset. Disabled by default unless ALLOW_DATA_SOURCE_OVERRIDE=1.", "example": "data/refined/future_flights.parquet"},
            ],
            "response_schema": {"type": "object"},
            "response_example": {
                "source": "file",
                "total_rows": 1384,
                "matched_rows": 214,
                "returned_rows": 50,
                "future_window": True,
                "predictions": [],
            },
            "deprecated": True,
        },
        {
            "tag": "Flights",
            "method": "GET",
            "path": "/api/flight/countries",
            "summary": "Available countries",
            "description": "Lists the countries present in the airport index to populate the frontend dropdowns.",
            "parameters": [
                {"name": "source_uri", "in": "query", "required": False, "schema": {"type": "string"}, "description": "Administrative source override for the airport index. Disabled by default unless ALLOW_DATA_SOURCE_OVERRIDE=1.", "example": "data/refined/world_airports.csv"},
            ],
            "response_schema": {"type": "object"},
            "response_example": {
                "source": "file",
                "total_countries": 3,
                "countries": [
                    {"country": "Brazil", "airport_count": 142},
                    {"country": "Portugal", "airport_count": 10},
                    {"country": "United States", "airport_count": 380},
                ],
            },
        },
        {
            "tag": "Flights",
            "method": "GET",
            "path": "/api/flight/airports",
            "summary": "Airports by country",
            "description": "Returns airports filtered by country, including city, name, and coordinates for the map.",
            "parameters": [
                {"name": "country", "in": "query", "required": True, "schema": {"type": "string"}, "description": "Country used as the main filter.", "example": "Brazil"},
                {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer", "default": 800, "minimum": 1, "maximum": 5000}, "description": "Maximum number of returned airports.", "example": 200},
                {"name": "source_uri", "in": "query", "required": False, "schema": {"type": "string"}, "description": "Administrative source override for the airport index. Disabled by default unless ALLOW_DATA_SOURCE_OVERRIDE=1.", "example": "data/refined/world_airports.csv"},
            ],
            "response_schema": {"type": "object"},
            "response_example": {
                "source": "file",
                "country": "Brazil",
                "total_airports": 2,
                "airports": [
                    {"iata_code": "GRU", "airport_name": "Sao Paulo/Guarulhos", "city": "Sao Paulo", "state": "SP", "country": "Brazil", "latitude": -23.4356, "longitude": -46.4731},
                    {"iata_code": "GIG", "airport_name": "Rio de Janeiro/Galeao", "city": "Rio de Janeiro", "state": "RJ", "country": "Brazil", "latitude": -22.8099, "longitude": -43.2506},
                ],
            },
            "error_responses": [
                {"status": 400, "description": "Missing country or invalid limit.", "example": {"detail": "Query parameter 'country' is required."}},
            ],
        },
        {
            "tag": "Flights",
            "method": "GET",
            "path": "/api/flight/departures",
            "summary": "Airport departures",
            "description": "Lists scheduled departures for a specific IATA airport with fallback to samples when no future source is available.",
            "parameters": [
                {"name": "airport", "in": "query", "required": True, "schema": {"type": "string"}, "description": "IATA code of the origin airport.", "example": "GRU"},
                {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500}, "description": "Maximum number of departures.", "example": 30},
                {"name": "source_uri", "in": "query", "required": False, "schema": {"type": "string"}, "description": "Administrative source override for the future-flights dataset. Disabled by default unless ALLOW_DATA_SOURCE_OVERRIDE=1.", "example": "data/refined/future_flights.parquet"},
            ],
            "response_schema": {"type": "object"},
            "response_example": {
                "source": "file",
                "airport": "GRU",
                "total_rows": 1384,
                "matched_rows": 42,
                "returned_rows": 30,
                "future_window": True,
                "departures": [{
                    "flight_date": "2026-04-10",
                    "flight_date_br": "10/04/2026",
                    "airline": "TP",
                    "flight_number": "88",
                    "flight_code": "TP88",
                    "origin_airport": "GRU",
                    "destination_airport": "LIS",
                    "scheduled_departure": "22:00",
                    "scheduled_arrival": "11:15",
                }],
            },
            "error_responses": [
                {"status": 400, "description": "Missing airport or invalid limit.", "example": {"detail": "Query parameter 'airport' is required."}},
            ],
        },
        {
            "tag": "Live Flights",
            "method": "GET",
            "path": "/api/live_flights",
            "summary": "Live flights by region or bounding box",
            "description": "Queries in-flight aircraft through OpenSky Network. Accepts a predefined region or a custom bounding box.",
            "parameters": [
                {"name": "region", "in": "query", "required": False, "schema": {"type": "string", "default": "brazil"}, "description": "Preconfigured region: brazil, south_america, north_america, europe, or world.", "example": "brazil"},
                {"name": "lamin", "in": "query", "required": False, "schema": {"type": "number"}, "description": "Minimum latitude of the bounding box.", "example": -23.7},
                {"name": "lomin", "in": "query", "required": False, "schema": {"type": "number"}, "description": "Minimum longitude of the bounding box.", "example": -46.9},
                {"name": "lamax", "in": "query", "required": False, "schema": {"type": "number"}, "description": "Maximum latitude of the bounding box.", "example": -23.4},
                {"name": "lomax", "in": "query", "required": False, "schema": {"type": "number"}, "description": "Maximum longitude of the bounding box.", "example": -46.5},
                {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer", "default": 200, "minimum": 1, "maximum": 500}, "description": "Maximum number of returned aircraft.", "example": 100},
                {"name": "include_ground", "in": "query", "required": False, "schema": {"type": "boolean", "default": False}, "description": "Includes aircraft on the ground when true.", "example": False},
            ],
            "response_schema": {"type": "object"},
            "response_example": {
                "source": "OpenSky Network",
                "region": "brazil",
                "bounding_box": {"lamin": -33.75, "lomin": -73.99, "lamax": 5.27, "lomax": -28.84},
                "include_ground": False,
                "cache_age_sec": 8,
                "total_found": 126,
                "returned": 100,
                "flights": [{"icao24": "e48f7f", "callsign": "TAM8085", "latitude": -23.31, "longitude": -46.11}],
            },
            "error_responses": [
                {"status": 400, "description": "Invalid bounding box or invalid limit.", "example": {"detail": "lamin/lomin/lamax/lomax must be decimal numbers."}},
                {"status": 502, "description": "OpenSky provider error.", "example": {"detail": "OpenSky returned an invalid response."}},
                {"status": 504, "description": "Timeout while querying the provider.", "example": {"detail": "OpenSky request timed out."}},
            ],
        },
        {
            "tag": "Live Flights",
            "method": "GET",
            "path": "/api/live_flights/{icao24}",
            "summary": "Live aircraft detail",
            "description": "Returns real-time data for a specific aircraft by ICAO24 code.",
            "parameters": [
                {"name": "icao24", "in": "path", "required": True, "schema": {"type": "string"}, "description": "Hexadecimal ICAO24 aircraft identifier.", "example": "e48f7f"},
            ],
            "response_schema": {"type": "object"},
            "response_example": {
                "source": "OpenSky Network",
                "cache_age_sec": 8,
                "flight": {"icao24": "e48f7f", "callsign": "TAM8085", "latitude": -23.31, "longitude": -46.11},
            },
            "error_responses": [
                {"status": 404, "description": "Aircraft not found in the current window.", "example": {"detail": "Aircraft 'e48f7f' not found or not currently in flight."}},
                {"status": 502, "description": "OpenSky provider error.", "example": {"detail": "OpenSky returned an invalid response."}},
                {"status": 504, "description": "Timeout while querying the provider.", "example": {"detail": "OpenSky request timed out."}},
            ],
        },
    ]


DOCS_BASE_URL_PLACEHOLDER = "$BASE_URL"


def public_source_label(source: str | None) -> str | None:
    if not source:
        return None
    text = str(source).strip()
    if not text:
        return None
    if text.startswith("inline:"):
        return text
    if is_s3_uri(text):
        return "s3"
    if text.startswith(("http://", "https://")):
        return "remote"
    return "file"


def internal_error_response(message: str, exc: Exception | None = None, status: int = 500):
    if exc is not None:
        if status >= 500:
            app.logger.exception("%s", message)
        else:
            app.logger.warning("%s: %s", message, exc)
    return jsonify({"detail": message}), status


def build_sample_url(base_url: str, endpoint: dict[str, Any]) -> str:
    path = endpoint["path"]
    query_params: dict[str, str] = {}
    for parameter in endpoint.get("parameters", []):
        example = parameter.get("example")
        if example is None:
            continue
        if parameter["in"] == "path":
            path = path.replace("{" + parameter["name"] + "}", str(example))
        elif parameter["in"] == "query":
            query_params[parameter["name"]] = str(example).lower() if isinstance(example, bool) else str(example)
    query = urlencode(query_params)
    return f"{base_url}{path}" + (f"?{query}" if query else "")


def build_curl_example(base_url: str, endpoint: dict[str, Any]) -> str:
    method = endpoint["method"].upper()
    url = build_sample_url(base_url, endpoint)
    if method == "GET":
        return f'curl -X GET "{url}"'
    request_example = endpoint.get("request_example")
    if request_example is None:
        return f'curl -X {method} "{url}"'
    body = json.dumps(request_example, ensure_ascii=False)
    return (
        f'curl -X {method} "{url}" \\\n'
        '  -H "Content-Type: application/json" \\\n'
        f"  -d '{body}'"
    )


def build_docs_sections(base_url: str) -> list[dict[str, Any]]:
    order = ["Meta", "Advisor", "Prediction", "Flights", "Live Flights"]
    catalog = build_api_docs_catalog()
    sections: list[dict[str, Any]] = []
    for tag in order:
        endpoints: list[dict[str, Any]] = []
        for endpoint in catalog:
            if endpoint["tag"] != tag:
                continue
            view = dict(endpoint)
            view["anchor"] = f'{endpoint["method"].lower()}-{endpoint["path"].strip("/").replace("/", "-").replace("{", "").replace("}", "") or "root"}'
            view["request_example_text"] = (
                json.dumps(endpoint["request_example"], ensure_ascii=False, indent=2)
                if endpoint.get("request_example") is not None else None
            )
            view["response_example_text"] = json.dumps(endpoint["response_example"], ensure_ascii=False, indent=2)
            view["curl_example"] = build_curl_example(base_url, endpoint)
            view["sample_url"] = build_sample_url(base_url, endpoint)
            view["error_responses"] = [
                {**item, "example_text": json.dumps(item["example"], ensure_ascii=False, indent=2)}
                for item in endpoint.get("error_responses", [])
            ]
            endpoints.append(view)
        sections.append({"tag": tag, "endpoints": endpoints})
    return sections


def build_openapi_spec(base_url: str) -> dict[str, Any]:
    paths: dict[str, Any] = {}
    for endpoint in build_api_docs_catalog():
        method = endpoint["method"].lower()
        operation: dict[str, Any] = {
            "tags": [endpoint["tag"]],
            "summary": endpoint["summary"],
            "description": endpoint["description"],
            "operationId": endpoint["path"].strip("/").replace("/", "_").replace("{", "").replace("}", "") or "root",
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": endpoint.get("response_schema", {"type": "object"}),
                            "example": endpoint["response_example"],
                        },
                    },
                },
            },
        }
        if endpoint.get("deprecated"):
            operation["deprecated"] = True
        if endpoint.get("parameters"):
            operation["parameters"] = endpoint["parameters"]
        if endpoint.get("request_schema") is not None:
            operation["requestBody"] = {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": endpoint["request_schema"],
                        "example": endpoint.get("request_example", {}),
                    },
                },
            }
        for error in endpoint.get("error_responses", []):
            operation["responses"][str(error["status"])] = {
                "description": error["description"],
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": error["example"],
                    },
                },
            }
        paths.setdefault(endpoint["path"], {})[method] = operation

    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Flight Advisor (Flask)",
            "version": "0.1.0",
            "description": "Flight Advisor HTTP API, including delay prediction, conversational advisor flows, future flights, and OpenSky integration.",
        },
        "servers": [{"url": base_url}],
        "tags": [{"name": name} for name in ["Meta", "Advisor", "Prediction", "Flights", "Live Flights"]],
        "paths": paths,
        "components": build_openapi_components(),
    }


# ── Page routes ──────────────────────────────────────────────────────────────

register_page_views(app)
register_advisor_views(app, {
    "AdviseRequest": AdviseRequest,
    "AdviseResponse": AdviseResponse,
    "ValidationError": ValidationError,
    "coerce_feature_types": coerce_feature_types,
    "load_advisor_messages": load_advisor_messages,
    "persist_advisor_messages": persist_advisor_messages,
    "reset_advisor_messages": reset_advisor_messages,
    "trim_chat_text": trim_chat_text,
    "should_run_route_prediction": should_run_route_prediction,
    "should_run_weekly_route_prediction": should_run_weekly_route_prediction,
    "enrich_route_payload_from_question": enrich_route_payload_from_question,
    "route_updates_from_request": route_updates_from_request,
    "user_chat_message": user_chat_message,
    "build_discovery_response_runtime": build_discovery_response_runtime,
    "should_use_llm": should_use_llm,
    "generate_llm_advice": generate_llm_advice,
    "build_discovery_context": build_discovery_context,
    "assistant_chat_message": assistant_chat_message,
    "load_assets": load_assets,
    "build_features": build_features,
    "compute_model_delay_snapshot": compute_model_delay_snapshot,
    "build_weekly_route_prediction": build_weekly_route_prediction,
    "risk_level": risk_level,
    "compute_top_factors": compute_top_factors,
    "build_suggested_flights": build_suggested_flights,
    "advice_text": advice_text,
    "build_advice_context": build_advice_context,
})
register_flight_views(app, {
    "load_airports_index": load_airports_index,
    "optional_text": optional_text,
    "load_upcoming_flights_frame": load_upcoming_flights_frame,
    "extract_airport_departures": extract_airport_departures,
})


# ── API routes ───────────────────────────────────────────────────────────────

@app.get("/api/routes")
def api_routes():
    return jsonify({
        "service": "Flight Advisor (Flask)", "docs": "/docs/", "redoc": "/redoc/",
        "openapi": "/openapi.json", "routes": list_route_index(),
        "examples": {
            "health":           {"method":"GET",  "path":"/health"},
            "advise":           {"method":"POST", "path":"/advise"},
            "advisor_history":  {"method":"GET",  "path":"/api/advisor/history"},
            "advisor_reset":    {"method":"POST", "path":"/api/advisor/reset"},
            "predict":          {"method":"POST", "path":"/predict"},
            "upcoming_flights": {"method":"GET",  "path":"/api/upcoming_flights?limit=50"},
            "flight_countries": {"method":"GET",  "path":"/api/flight/countries"},
            "flight_airports":  {"method":"GET",  "path":"/api/flight/airports?country=Brazil"},
            "flight_departures":{"method":"GET",  "path":"/api/flight/departures?airport=GRU&limit=30"},
        },
    })

@app.route("/docs", methods=["GET"], strict_slashes=False)
def docs():
    sections = build_docs_sections(DOCS_BASE_URL_PLACEHOLDER)
    endpoint_count = sum(len(section["endpoints"]) for section in sections)
    return render_template(
        "api_docs.html",
        page_title="Flight Advisor | API Docs",
        active_page="docs",
        doc_sections=sections,
        docs_base_url=DOCS_BASE_URL_PLACEHOLDER,
        openapi_url=url_for("openapi_json"),
        routes_url=url_for("api_routes"),
        endpoint_count=endpoint_count,
        post_count=sum(1 for section in sections for endpoint in section["endpoints"] if endpoint["method"] == "POST"),
        get_count=sum(1 for section in sections for endpoint in section["endpoints"] if endpoint["method"] == "GET"),
    )

@app.route("/redoc", methods=["GET"], strict_slashes=False)
def redoc():
    return redirect(url_for("docs"))

@app.get("/openapi.json")
def openapi_json():
    return jsonify(build_openapi_spec("/"))

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.get("/api/upcoming_flights")
@app.get("/api/weekly_predictions")
def upcoming_flights():
    limit_raw  = request.args.get("limit","50")
    source_uri = request.args.get("source_uri")
    try: limit = max(1, min(int(limit_raw), 500))
    except ValueError: return jsonify({"detail":"Query parameter 'limit' must be an integer."}), 400
    try: df, source = load_upcoming_flights_frame(source_uri)
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400
    except FileNotFoundError:
        return jsonify({"source":None,"total_rows":0,"matched_rows":0,"returned_rows":0,
                        "future_window":False,"predictions":[],"detail":"Upcoming flights dataset is unavailable."})
    except Exception as exc:
        return internal_error_response("Unable to load upcoming flights right now.", exc)
    flights, matched, future = extract_upcoming_flights(df, limit)
    return jsonify({"source":public_source_label(source),"total_rows":len(df),"matched_rows":matched,
                    "returned_rows":len(flights),"future_window":future,"predictions":flights})

@app.route("/predict", methods=["POST","OPTIONS"])
def predict():
    if request.method == "OPTIONS": return ("", 204)
    payload_data = request.get_json(silent=True)
    if payload_data is None:
        if request.get_data(as_text=True).strip(): return jsonify({"detail":"Body must be valid JSON."}), 400
        payload_data = {}
    try: payload = PredictRequest.model_validate(payload_data)
    except ValidationError as exc: return jsonify({"detail":exc.errors()}), 400
    try:
        pipeline, meta, *_ = load_assets()
    except Exception as exc:
        return internal_error_response("Prediction assets are unavailable right now.", exc)
    try: base_df, source = load_predict_frame(payload)
    except ValueError as exc:
        return jsonify({"detail":str(exc)}), 400
    except FileNotFoundError:
        return jsonify({"detail":"Prediction input dataset is unavailable."}), 400
    except Exception as exc:
        return internal_error_response("Unable to load prediction input right now.", exc)
    total_rows = int(len(base_df))
    if total_rows == 0:
        return jsonify({"source":public_source_label(source),"threshold":payload.threshold,"total_rows":0,
                        "matched_rows":0,"returned_rows":0,"predictions":[]})
    try: filtered_df = apply_predict_filters(base_df, payload)
    except ValueError as exc: return jsonify({"detail":str(exc)}), 400
    matched = int(len(filtered_df))
    if matched == 0:
        return jsonify({"source":public_source_label(source),"threshold":payload.threshold,"total_rows":total_rows,
                        "matched_rows":0,"returned_rows":0,"predictions":[]})
    try: output_df = predict_dataframe(filtered_df.head(payload.limit).copy(), pipeline, meta, payload.threshold)
    except ValueError as exc: return jsonify({"detail":str(exc)}), 400
    except Exception as exc:
        return internal_error_response("Unable to compute predictions right now.", exc)
    return jsonify({"source":public_source_label(source),"threshold":payload.threshold,"total_rows":total_rows,
                    "matched_rows":matched,"returned_rows":len(output_df),"predictions":dataframe_json_records(output_df)})


def _runtime_host() -> str:
    host = (os.getenv("API_HOST") or os.getenv("HOST") or "").strip()
    if host:
        return host
    return "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"


def _runtime_port(default: int = 8000) -> int:
    raw = (os.getenv("PORT") or os.getenv("API_PORT") or str(default)).strip()
    try:
        port = int(raw)
    except ValueError:
        print(f"Invalid port '{raw}'. Falling back to {default}.", file=sys.stderr)
        return default
    return max(1, min(port, 65535))


def _runtime_debug() -> bool:
    raw = (os.getenv("FLASK_DEBUG") or os.getenv("DEBUG") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def run_local_server() -> int:
    app.run(host=_runtime_host(), port=_runtime_port(), debug=_runtime_debug())
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
        return internal_error_response("OpenSky request timed out.", exc, status=504)
    except RuntimeError as exc:
        return internal_error_response("OpenSky provider error.", exc, status=502)
    except Exception as exc:
        return internal_error_response("Unexpected live-flight lookup error.", exc)

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
        return internal_error_response("OpenSky request timed out.", exc, status=504)
    except RuntimeError as exc:
        return internal_error_response("OpenSky provider error.", exc, status=502)
    except Exception as exc:
        return internal_error_response("Unexpected live-flight detail error.", exc)

    match = next((f for f in flights if f.get("icao24", "").lower() == icao24), None)
    if not match:
        return jsonify({"detail": f"Aircraft '{icao24}' not found or not currently in flight."}), 404

    return jsonify({"source": "OpenSky Network", "cache_age_sec": cache_age, "flight": match})


if __name__ == "__main__":
    raise SystemExit(run_local_server())
