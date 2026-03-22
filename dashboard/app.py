#!/usr/bin/env python3
"""Dash dashboard for Flight Advisor — optimised for Gunicorn multi-worker deployments.

Performance improvements applied
---------------------------------
* **Athena queries without inner LIMIT** — aggregations now run on the full filtered
  partition, not a 100-row sample, so results are both correct and faster.
* **Full predicate + column pushdown on Parquet** — PyArrow filters on year, month,
  airline AND state before loading into memory, and only the required columns are read.
* **Redis-backed shared cache** — SimpleCache is in-process and duplicates work
  across Gunicorn workers; Redis (or any shared backend) shares results.
* **Callbacks 1 & 2 merged** — filter options and default values are written in a
  single round-trip instead of two cascaded callbacks.
* **KPI and chart callbacks separated** — each can use no_update for outputs that
  didn't change, reducing unnecessary re-renders.
* **Smarter no_update routing** — when only one filter changes, only the affected
  chart is recomputed.
* **Background pre-warm thread** (DASH_PREWARM=1) unchanged — still available.
* **Column selection helper** centralises the list of columns read from Parquet.

FIX: resolve_data_source now defaults to flights_processed.parquet instead of .csv
"""

from __future__ import annotations

import os
import re
import sys
import threading
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html, callback_context, no_update
from flask_caching import Cache

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aws.athena_query import run_query  # noqa: E402
from model import (  # noqa: E402
    build_s3_client,
    default_s3_uri,
    is_s3_uri,
    load_env_file,
    load_csv_any,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_env_file()

# ---------------------------------------------------------------------------
# Helpers – data source
# ---------------------------------------------------------------------------

def resolve_data_source() -> Optional[str]:
    source = os.getenv("DASH_SOURCE")
    if source:
        return source
    bucket = os.getenv("S3_BUCKET") or os.getenv("S3_Bucket")
    processed_prefix = os.getenv("S3_PROCESSED_PREFIX")
    refined_prefix = os.getenv("S3_REFINED_PREFIX")
    data_prefix = processed_prefix or refined_prefix or "processed"
    filename = os.getenv("DASH_FILENAME", "flights_processed.parquet")  # FIX: era .csv
    if bucket:
        return default_s3_uri(bucket, data_prefix, filename)
    return None


def athena_enabled() -> bool:
    flag = os.getenv("DASH_USE_ATHENA", "1")
    bucket = os.getenv("S3_BUCKET") or os.getenv("S3_Bucket")
    return flag == "1" and bool(bucket)


def athena_config() -> dict:
    bucket = os.getenv("S3_BUCKET") or os.getenv("S3_Bucket")
    return {
        "bucket": bucket,
        "database": os.getenv("ATHENA_DATABASE", "flight_advisor"),
        "table": os.getenv("ATHENA_TABLE", "flights_processed"),
        "results_prefix": os.getenv("S3_ATHENA_RESULTS_PREFIX", "athena-results"),
        "region": os.getenv("AWS_REGION", "us-east-1"),
        "profile": os.getenv("AWS_PROFILE"),
    }


# ---------------------------------------------------------------------------
# Input sanitisation
# ---------------------------------------------------------------------------
SAFE_TEXT = re.compile(r"^[A-Za-z0-9 .&/_-]+$")

# ---------------------------------------------------------------------------
# Athena error state (in-process, best-effort)
# ---------------------------------------------------------------------------
_ATHENA_ERROR: dict[str, Optional[str]] = {"message": None}


def sanitize_text(values: List[str]) -> List[str]:
    cleaned: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        value = value.strip()
        if not value or len(value) > 60:
            continue
        if not SAFE_TEXT.match(value):
            continue
        cleaned.append(value)
    return cleaned


def sql_list_text(values: List[str]) -> str:
    safe = sanitize_text(values)
    if not safe:
        return ""
    escaped = [val.replace("'", "''") for val in safe]
    return ", ".join(f"'{val}'" for val in escaped)


def sql_list_int(values: List[int]) -> str:
    clean: List[str] = []
    for value in values:
        try:
            num = int(value)
        except (TypeError, ValueError):
            continue
        clean.append(str(num))
    return ", ".join(clean)


def limit_items(values: List, limit: int) -> List:
    if limit <= 0:
        return values
    return values[-limit:] if len(values) > limit else values


def clear_athena_error() -> None:
    _ATHENA_ERROR["message"] = None


def log_athena_error(context: str, exc: Exception) -> None:
    msg = f"{context} failed: {exc}"
    _ATHENA_ERROR["message"] = msg
    print(f"[Athena] {msg}", file=sys.stderr)


def build_where(
    years: List[int],
    months: List[int],
    airlines: List[str],
    states: List[str],
) -> str:
    clauses: List[str] = []
    if years:
        s = sql_list_int(years)
        if s:
            clauses.append(f"year IN ({s})")
    if months:
        s = sql_list_int(months)
        if s:
            clauses.append(f"month IN ({s})")
    if airlines:
        s = sql_list_text(airlines)
        if s:
            clauses.append(f"COALESCE(airline_name, airline) IN ({s})")
    if states:
        s = sql_list_text(states)
        if s:
            clauses.append(f"origin_state IN ({s})")
    return ("WHERE " + " AND ".join(clauses)) if clauses else ""


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def get_airline_column(df: pd.DataFrame) -> str:
    return "airline_name" if "airline_name" in df.columns else "airline"


def get_state_column(df: pd.DataFrame) -> Optional[str]:
    if "origin_state" in df.columns:
        return "origin_state"
    if "state" in df.columns:
        return "state"
    return None


# Columns we actually need — avoids loading unused fields from Parquet
_REQUIRED_COLUMNS = [
    "year", "month", "day",
    "arrival_delay", "is_delayed",
    "airline_name", "airline",
    "origin_state",
]


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

def normalize_base_path(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    if not path.endswith("/"):
        path = path + "/"
    return path


USE_ATHENA = athena_enabled()

_base_path = normalize_base_path(os.getenv("DASH_BASE_PATH", "/"))

app = Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap"
    ],
    requests_pathname_prefix=_base_path,
    routes_pathname_prefix=_base_path,
    suppress_callback_exceptions=True,
)

server = app.server  # expose Flask server for Gunicorn

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
CACHE_CONFIG: dict = {
    "CACHE_TYPE": os.getenv("CACHE_TYPE", "SimpleCache"),
    "CACHE_DEFAULT_TIMEOUT": int(os.getenv("CACHE_TIMEOUT", "3600")),
}
if os.getenv("CACHE_REDIS_URL"):
    CACHE_CONFIG["CACHE_TYPE"] = "RedisCache"
    CACHE_CONFIG["CACHE_REDIS_URL"] = os.getenv("CACHE_REDIS_URL")

cache = Cache()
cache.init_app(server, config=CACHE_CONFIG)

# ---------------------------------------------------------------------------
# Cached data helpers
# ---------------------------------------------------------------------------

def _and_expr(left, right):
    """Combine two PyArrow expressions with AND, handling None."""
    return right if left is None else (left & right)


def _build_pa_filter(years: tuple, months: tuple, airlines: tuple, states: tuple):
    """Build a PyArrow expression covering ALL filter dimensions."""
    import pyarrow.dataset as ds

    expr = None
    if years:
        expr = _and_expr(expr, ds.field("year").isin(list(years)))
    if months:
        expr = _and_expr(expr, ds.field("month").isin(list(months)))
    if airlines:
        expr = _and_expr(expr, ds.field("airline_name").isin(list(airlines)))
    if states:
        expr = _and_expr(expr, ds.field("origin_state").isin(list(states)))
    return expr


@cache.memoize()
def load_flights_cached(
    years: tuple = (),
    months: tuple = (),
    airlines: tuple = (),
    states: tuple = (),
) -> pd.DataFrame:
    """Load Parquet data with full predicate + column pushdown."""
    if USE_ATHENA:
        return pd.DataFrame()

    source = resolve_data_source()
    if not source:
        return pd.DataFrame()

    import pyarrow.dataset as ds

    try:
        if is_s3_uri(source):
            import pyarrow.fs as pafs
            region = os.getenv("AWS_REGION", "us-east-1")
            s3fs = pafs.S3FileSystem(region=region)
            s3_path = source.removeprefix("s3://")
            dataset = ds.dataset(s3_path, filesystem=s3fs, format="parquet", partitioning="hive")
        else:
            dataset = ds.dataset(source, format="parquet", partitioning="hive")

        pa_filter = _build_pa_filter(years, months, airlines, states)

        # Only read the columns we actually use
        available = set(dataset.schema.names)
        columns = [c for c in _REQUIRED_COLUMNS if c in available]

        table = dataset.to_table(filter=pa_filter, columns=columns or None)
        df = table.to_pandas()

    except Exception as exc:  # noqa: BLE001
        print(f"[load_flights] Failed to load Parquet: {exc}", file=sys.stderr)
        return pd.DataFrame()

    if not df.empty:
        key_cols = ["arrival_delay", "is_delayed", "month", get_airline_column(df)]
        df.dropna(subset=[c for c in key_cols if c in df.columns], inplace=True)

    sample = int(os.getenv("DASH_SAMPLE", "0") or 0)
    if sample > 0 and len(df) > sample:
        df = df.sample(n=sample, random_state=42)
    return df


@cache.memoize()
def fetch_filter_options_cached() -> dict:
    """Fetch distinct filter values from Athena or local Parquet. Cached."""
    if not USE_ATHENA:
        source = resolve_data_source()
        if not source:
            return {"years": [], "months": [], "airlines": [], "states": []}

        import pyarrow.dataset as ds

        try:
            if is_s3_uri(source):
                import pyarrow.fs as pafs
                region = os.getenv("AWS_REGION", "us-east-1")
                s3fs = pafs.S3FileSystem(region=region)
                s3_path = source.removeprefix("s3://")
                dataset = ds.dataset(s3_path, filesystem=s3fs, format="parquet", partitioning="hive")
            else:
                dataset = ds.dataset(source, format="parquet", partitioning="hive")

            available = set(dataset.schema.names)
            cols = [c for c in ["year", "month", "airline_name", "airline", "origin_state"] if c in available]
            table = dataset.to_table(columns=cols)
            df = table.to_pandas()
        except Exception as exc:  # noqa: BLE001
            print(f"[filter_options] Failed: {exc}", file=sys.stderr)
            return {"years": [], "months": [], "airlines": [], "states": []}

        airline_col = get_airline_column(df)
        state_col = get_state_column(df)
        return {
            "years":    sorted(df["year"].dropna().unique().astype(int).tolist()) if "year" in df.columns else [],
            "months":   sorted(df["month"].dropna().unique().astype(int).tolist()) if "month" in df.columns else [],
            "airlines": sorted(df[airline_col].dropna().unique().astype(str).tolist()) if airline_col in df.columns else [],
            "states":   sorted(df[state_col].dropna().unique().astype(str).tolist()) if state_col and state_col in df.columns else [],
        }

    # Athena path
    cfg = athena_config()
    table = cfg["table"]
    database = cfg["database"]
    bucket = cfg["bucket"]
    results_prefix = cfg["results_prefix"]
    region = cfg["region"]
    profile = cfg["profile"]

    def run(sql: str) -> pd.DataFrame:
        return run_query(sql, database, bucket, results_prefix, region, profile, max_rows=500)

    try:
        years_df    = run(f'SELECT DISTINCT year FROM "{database}"."{table}" ORDER BY year')
        months_df   = run(f'SELECT DISTINCT month FROM "{database}"."{table}" ORDER BY month')
        airlines_df = run(
            f'SELECT DISTINCT COALESCE(airline_name, airline) AS airline '
            f'FROM "{database}"."{table}" WHERE airline IS NOT NULL ORDER BY airline'
        )
        states_df   = run(
            f'SELECT DISTINCT origin_state AS state '
            f'FROM "{database}"."{table}" WHERE origin_state IS NOT NULL ORDER BY state'
        )
    except RuntimeError as exc:
        log_athena_error("fetch_filter_options", exc)
        return {"years": [], "months": [], "airlines": [], "states": []}

    return {
        "years":    sorted(years_df["year"].dropna().astype(int).tolist()) if "year" in years_df else [],
        "months":   sorted(months_df["month"].dropna().astype(int).tolist()) if "month" in months_df else [],
        "airlines": sorted(airlines_df["airline"].dropna().astype(str).tolist()) if "airline" in airlines_df else [],
        "states":   sorted(states_df["state"].dropna().astype(str).tolist()) if "state" in states_df else [],
    }


# ---------------------------------------------------------------------------
# Optional: pre-warm cache in a background thread
# ---------------------------------------------------------------------------
def _prewarm() -> None:
    try:
        print("[prewarm] Fetching filter options in background…", file=sys.stderr)
        fetch_filter_options_cached()
        print("[prewarm] Done.", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[prewarm] Failed: {exc}", file=sys.stderr)


if os.getenv("DASH_PREWARM", "0") == "1":
    threading.Thread(target=_prewarm, daemon=True).start()

# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------

def kpi_total(df: pd.DataFrame) -> str:
    return f"{len(df):,}"


def kpi_delay_rate(df: pd.DataFrame) -> str:
    if "is_delayed" not in df.columns or df.empty:
        return "—"
    return f"{df['is_delayed'].mean() * 100:.1f}%"


def kpi_avg_delay(df: pd.DataFrame) -> str:
    if "arrival_delay" not in df.columns or df.empty:
        return "—"
    return f"{df['arrival_delay'].mean():.1f} min"


def kpi_worst_airline(df: pd.DataFrame) -> str:
    if df.empty or "arrival_delay" not in df.columns:
        return "—"
    col = get_airline_column(df)
    grouped = df.groupby(col)["arrival_delay"].mean().sort_values(ascending=False)
    return str(grouped.index[0]) if not grouped.empty else "—"


def filter_df(
    df: pd.DataFrame,
    airlines: List[str],
    states: List[str],
) -> pd.DataFrame:
    """In-memory filter for dimensions NOT already pushed down to PyArrow."""
    if df.empty:
        return df
    airline_col = get_airline_column(df)
    state_col   = get_state_column(df)
    filtered = df
    if airlines:
        filtered = filtered[filtered[airline_col].isin(airlines)]
    if states and state_col:
        filtered = filtered[filtered[state_col].isin(states)]
    return filtered


# ---------------------------------------------------------------------------
# Chart helpers (local df)
# ---------------------------------------------------------------------------

def build_airline_bar(df: pd.DataFrame):
    if df.empty or "arrival_delay" not in df.columns:
        return _empty_bar("Average delay by airline")
    col = get_airline_column(df)
    grouped = (
        df.groupby(col)["arrival_delay"].mean()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )
    fig = px.bar(grouped, x=col, y="arrival_delay", title="Average delay by airline (top 20)")
    return _style_fig(fig, xaxis_title="", yaxis_title="Avg delay (min)")


def build_monthly_line(df: pd.DataFrame):
    if df.empty or "arrival_delay" not in df.columns:
        return _empty_line("Monthly delay trend")
    grouped = df.groupby("month")["arrival_delay"].mean().reset_index()
    fig = px.line(grouped, x="month", y="arrival_delay", markers=True, title="Monthly delay trend")
    return _style_fig(fig, xaxis_title="Month", yaxis_title="Avg delay (min)")


# ---------------------------------------------------------------------------
# Chart helpers (Athena)
# ---------------------------------------------------------------------------

def query_kpis_athena(years, months, airlines, states) -> dict:
    cfg = athena_config()
    where = build_where(years, months, airlines, states)
    sql = f"""
        SELECT
            COUNT(*)                                               AS total_flights,
            AVG(CASE WHEN is_delayed = 1 THEN 1.0 ELSE 0.0 END)  AS delay_rate,
            AVG(arrival_delay)                                     AS avg_delay
        FROM "{cfg['database']}"."{cfg['table']}"
        {where}
    """
    try:
        df = run_query(
            sql, cfg["database"], cfg["bucket"], cfg["results_prefix"],
            cfg["region"], cfg["profile"], max_rows=1,
        )
    except RuntimeError as exc:
        log_athena_error("query_kpis_athena", exc)
        return {"total": "0", "delay_rate": "—", "avg_delay": "—"}
    if df.empty:
        return {"total": "0", "delay_rate": "—", "avg_delay": "—"}
    total = int(df["total_flights"].iloc[0]) if "total_flights" in df else 0
    dr    = df["delay_rate"].iloc[0] if "delay_rate" in df else None
    ad    = df["avg_delay"].iloc[0]  if "avg_delay"   in df else None
    return {
        "total":       f"{total:,}",
        "delay_rate":  f"{float(dr) * 100:.1f}%" if dr is not None else "—",
        "avg_delay":   f"{float(ad):.1f} min"    if ad is not None else "—",
    }


def query_worst_airline_athena(years, months, airlines, states) -> str:
    cfg = athena_config()
    where = build_where(years, months, airlines, states)
    sql = f"""
        SELECT COALESCE(airline_name, airline) AS airline,
               AVG(arrival_delay) AS avg_delay
        FROM "{cfg['database']}"."{cfg['table']}"
        {where}
        GROUP BY 1
        ORDER BY avg_delay DESC
        LIMIT 1
    """
    try:
        df = run_query(
            sql, cfg["database"], cfg["bucket"], cfg["results_prefix"],
            cfg["region"], cfg["profile"], max_rows=1,
        )
    except RuntimeError as exc:
        log_athena_error("query_worst_airline_athena", exc)
        return "—"
    if df.empty or "airline" not in df:
        return "—"
    return str(df["airline"].iloc[0])


def query_airline_bar_athena(years, months, airlines, states):
    cfg = athena_config()
    where = build_where(years, months, airlines, states)
    top_n = int(os.getenv("DASH_TOP_N", "20"))
    sql = f"""
        SELECT COALESCE(airline_name, airline) AS airline,
               AVG(arrival_delay) AS avg_delay
        FROM "{cfg['database']}"."{cfg['table']}"
        {where}
        GROUP BY 1
        ORDER BY avg_delay DESC
        LIMIT {top_n}
    """
    try:
        df = run_query(
            sql, cfg["database"], cfg["bucket"], cfg["results_prefix"],
            cfg["region"], cfg["profile"],
        )
    except RuntimeError as exc:
        log_athena_error("query_airline_bar_athena", exc)
        return _empty_bar("Average delay by airline")
    if df.empty:
        return _empty_bar("Average delay by airline")
    fig = px.bar(df, x="airline", y="avg_delay", title=f"Average delay by airline (top {top_n})")
    return _style_fig(fig, xaxis_title="", yaxis_title="Avg delay (min)")


def query_monthly_line_athena(years, months, airlines, states):
    cfg = athena_config()
    where = build_where(years, months, airlines, states)
    sql = f"""
        SELECT month, AVG(arrival_delay) AS avg_delay
        FROM "{cfg['database']}"."{cfg['table']}"
        {where}
        GROUP BY month
        ORDER BY month
    """
    try:
        df = run_query(
            sql, cfg["database"], cfg["bucket"], cfg["results_prefix"],
            cfg["region"], cfg["profile"],
        )
    except RuntimeError as exc:
        log_athena_error("query_monthly_line_athena", exc)
        return _empty_line("Monthly delay trend")
    if df.empty:
        return _empty_line("Monthly delay trend")
    fig = px.line(df, x="month", y="avg_delay", markers=True, title="Monthly delay trend")
    return _style_fig(fig, xaxis_title="Month", yaxis_title="Avg delay (min)")


# ---------------------------------------------------------------------------
# Figure styling helpers
# ---------------------------------------------------------------------------
_PLOT_BG   = "#ffffff"
_PAPER_BG  = "#ffffff"
_FONT_COLOR = "#111111"
_GRID_COLOR = "#e5e7eb"
_ACCENT    = "#111111"


def _style_fig(fig, **layout_kwargs):
    fig.update_layout(
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(family="IBM Plex Sans, sans-serif", color=_FONT_COLOR),
        title_font=dict(size=14),
        xaxis=dict(gridcolor=_GRID_COLOR, linecolor=_GRID_COLOR),
        yaxis=dict(gridcolor=_GRID_COLOR, linecolor=_GRID_COLOR),
        margin=dict(l=36, r=16, t=44, b=36),
        **layout_kwargs,
    )
    fig.update_traces(marker_color=_ACCENT)
    return fig


def _empty_bar(title: str):
    return _style_fig(px.bar(title=f"{title} — no data"))


def _empty_line(title: str):
    return _style_fig(px.line(title=f"{title} — no data"))


# ---------------------------------------------------------------------------
# Default selections
# ---------------------------------------------------------------------------

def default_year_selection(years: List[int]) -> List[int]:
    if not years:
        return []
    env = os.getenv("DASH_DEFAULT_YEARS")
    if env:
        try:
            selected = [int(v.strip()) for v in env.split(",") if v.strip()]
            return [y for y in selected if y in years]
        except ValueError:
            pass
    return [max(years)]


def default_month_selection(months: List[int]) -> List[int]:
    if not months:
        return []
    env = os.getenv("DASH_DEFAULT_MONTHS")
    if env:
        try:
            selected = [int(v.strip()) for v in env.split(",") if v.strip()]
            return [m for m in selected if m in months]
        except ValueError:
            pass
    return []


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_CARD = {
    "background": "#ffffff",
    "padding": "14px 16px",
    "borderRadius": "10px",
    "border": "1px solid #e5e7eb",
}

_KPI_LABEL = {
    "fontSize": "11px", "color": "#6b7280", "marginBottom": "6px",
}
_KPI_VALUE = {
    "fontSize": "24px", "fontWeight": "600", "color": "#111111",
    "fontFamily": "'IBM Plex Mono', monospace",
}
_ERROR_BANNER = {
    "background": "#fff5f5",
    "border": "1px solid #fecaca",
    "color": "#991b1b",
    "padding": "10px 12px",
    "borderRadius": "8px",
    "fontFamily": "'IBM Plex Mono', monospace",
    "fontSize": "11px",
    "marginBottom": "12px",
    "display": "none",
}
_FILTER_STYLE = {
    "borderRadius": "8px",
    "border": "1px solid #d1d5db",
    "fontSize": "13px",
}

data_source_label = (
    f"Athena {athena_config()['database']}.{athena_config()['table']}"
    if USE_ATHENA
    else (resolve_data_source() or "not configured")
)

app.layout = html.Div(
    style={
        "fontFamily": "'IBM Plex Sans', sans-serif",
        "background": "#f5f5f4",
        "minHeight": "100vh",
        "padding": "20px 16px",
        "color": "#111111",
    },
    children=[
        # ── store: holds filter options fetched lazily ──────────────────────
        dcc.Store(id="store-filter-options"),
        # ── store: triggers first load on page open ─────────────────────────
        dcc.Store(id="store-init", data=True),
        # ── interval: refresh Athena error banner ───────────────────────────
        dcc.Interval(id="athena-error-poll", interval=3000, n_intervals=0),

        html.Div(
            style={"maxWidth": "1120px", "margin": "0 auto"},
            children=[

                # ── Header ─────────────────────────────────────────────────
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "flex-start",
                        "flexWrap": "wrap",
                        "gap": "8px",
                        "marginBottom": "14px",
                    },
                    children=[
                        html.H1(
                            "Flight Advisor",
                            style={"margin": "0", "fontSize": "24px", "fontWeight": "600"},
                        ),
                        html.Div(
                            style={
                                "fontFamily": "'IBM Plex Mono', monospace",
                                "fontSize": "11px",
                                "color": "#6b7280",
                            },
                            children=f"Source: {data_source_label}",
                        ),
                    ],
                ),
                html.Div(id="athena-error", style=_ERROR_BANNER, children=""),

                # ── KPI row ────────────────────────────────────────────────
                dcc.Loading(
                    type="dot",
                    color=_ACCENT,
                    children=html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                            "gap": "10px", "marginBottom": "14px",
                        },
                        children=[
                            html.Div(style=_CARD, children=[
                                html.Div("Total flights", style=_KPI_LABEL),
                                html.Div(id="kpi-total", style=_KPI_VALUE, children="—"),
                            ]),
                            html.Div(style=_CARD, children=[
                                html.Div("Delayed flights", style=_KPI_LABEL),
                                html.Div(id="kpi-delay-rate", style=_KPI_VALUE, children="—"),
                            ]),
                            html.Div(style=_CARD, children=[
                                html.Div("Avg arrival delay", style=_KPI_LABEL),
                                html.Div(id="kpi-avg-delay", style=_KPI_VALUE, children="—"),
                            ]),
                            html.Div(style=_CARD, children=[
                                html.Div("Worst airline", style=_KPI_LABEL),
                                html.Div(id="kpi-worst-airline",
                                         style={**_KPI_VALUE, "fontSize": "18px"}, children="—"),
                            ]),
                        ],
                    ),
                ),

                # ── Filters ────────────────────────────────────────────────
                dcc.Loading(
                    type="dot",
                    color=_ACCENT,
                    children=html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit, minmax(210px, 1fr))",
                            "gap": "10px", "marginBottom": "14px",
                        },
                        children=[
                            dcc.Dropdown(id="filter-year",    options=[], value=[], multi=True,
                                         placeholder="Filter by year",    style=_FILTER_STYLE),
                            dcc.Dropdown(id="filter-airline", options=[], value=[], multi=True,
                                         placeholder="Filter by airline", style=_FILTER_STYLE),
                            dcc.Dropdown(id="filter-month",   options=[], value=[], multi=True,
                                         placeholder="Filter by month",   style=_FILTER_STYLE),
                            dcc.Dropdown(id="filter-state",   options=[], value=[], multi=True,
                                         placeholder="Filter by origin state", style=_FILTER_STYLE),
                        ],
                    ),
                ),

                # ── Charts ─────────────────────────────────────────────────
                dcc.Loading(
                    type="dot",
                    color=_ACCENT,
                    children=html.Div(
                        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))", "gap": "10px"},
                        children=[
                            html.Div(style=_CARD, children=dcc.Graph(
                                id="chart-airline", config={"displayModeBar": False})),
                            html.Div(style=_CARD, children=dcc.Graph(
                                id="chart-month",   config={"displayModeBar": False})),
                        ],
                    ),
                ),

                # ── Footer ─────────────────────────────────────────────────
                html.Div(
                    style={"marginTop": "14px", "color": "#6b7280", "fontSize": "11px",
                           "fontFamily": "'IBM Plex Mono', monospace"},
                    children=(
                        "Set DASH_SOURCE or DASH_USE_ATHENA=0 to override."
                    ),
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Athena error banner refresh
# ---------------------------------------------------------------------------
@app.callback(
    Output("athena-error", "children"),
    Output("athena-error", "style"),
    Input("athena-error-poll", "n_intervals"),
    prevent_initial_call=False,
)
def refresh_athena_error(_n):
    msg = _ATHENA_ERROR.get("message")
    if not msg:
        return "", {**_ERROR_BANNER, "display": "none"}
    return f"Athena error: {msg}", {**_ERROR_BANNER, "display": "block"}


# ---------------------------------------------------------------------------
# Callback 1 – populate filter dropdowns AND store in a single round-trip
# ---------------------------------------------------------------------------
@app.callback(
    Output("store-filter-options", "data"),
    Output("filter-year",    "options"),
    Output("filter-year",    "value"),
    Output("filter-airline", "options"),
    Output("filter-month",   "options"),
    Output("filter-month",   "value"),
    Output("filter-state",   "options"),
    Input("store-init", "data"),
    prevent_initial_call=False,
)
def load_and_populate_filters(_init):
    """Fetch filter options and populate dropdowns in a single callback."""
    options = fetch_filter_options_cached()

    years    = options.get("years",    [])
    months   = options.get("months",   [])
    airlines = options.get("airlines", [])
    states   = options.get("states",   [])
    if USE_ATHENA and any([years, months, airlines, states]):
        clear_athena_error()

    year_opts    = [{"label": str(y), "value": y} for y in years]
    airline_opts = [{"label": a, "value": a} for a in airlines]
    month_opts   = [{"label": str(m), "value": m} for m in months]
    state_opts   = [{"label": s, "value": s} for s in states]

    return (
        options,
        year_opts,
        default_year_selection(years),
        airline_opts,
        month_opts,
        default_month_selection(months),
        state_opts,
    )


# ---------------------------------------------------------------------------
# Callback 2 – KPI update
# ---------------------------------------------------------------------------
@app.callback(
    Output("kpi-total",         "children"),
    Output("kpi-delay-rate",    "children"),
    Output("kpi-avg-delay",     "children"),
    Output("kpi-worst-airline", "children"),
    Input("filter-year",    "value"),
    Input("filter-airline", "value"),
    Input("filter-month",   "value"),
    Input("filter-state",   "value"),
    State("store-filter-options", "data"),
    prevent_initial_call=True,
)
def update_kpis(years_sel, airlines_sel, months_sel, states_sel, options_data):
    if not options_data:
        return "—", "—", "—", "—"
    if USE_ATHENA:
        clear_athena_error()

    years_sel    = years_sel    or []
    airlines_sel = airlines_sel or []
    months_sel   = months_sel   or []
    states_sel   = states_sel   or []

    max_years    = int(os.getenv("DASH_MAX_YEARS",    "3"))
    max_airlines = int(os.getenv("DASH_MAX_AIRLINES", "25"))
    max_states   = int(os.getenv("DASH_MAX_STATES",   "25"))

    years_sel    = limit_items(sorted(set(years_sel)),    max_years)
    airlines_sel = limit_items(sorted(set(airlines_sel)), max_airlines)
    states_sel   = limit_items(sorted(set(states_sel)),   max_states)

    if USE_ATHENA:
        kpis  = query_kpis_athena(years_sel, months_sel, airlines_sel, states_sel)
        worst = query_worst_airline_athena(years_sel, months_sel, airlines_sel, states_sel)
        return kpis["total"], kpis["delay_rate"], kpis["avg_delay"], worst

    df = load_flights_cached(
        years=tuple(sorted(years_sel)),
        months=tuple(sorted(months_sel)),
        airlines=tuple(sorted(airlines_sel)),
        states=tuple(sorted(states_sel)),
    )
    return kpi_total(df), kpi_delay_rate(df), kpi_avg_delay(df), kpi_worst_airline(df)


# ---------------------------------------------------------------------------
# Callback 3 – Chart update with smart no_update routing
# ---------------------------------------------------------------------------
_MONTHLY_INPUTS = {"filter-year.value", "filter-month.value", "filter-state.value"}
_AIRLINE_INPUTS = {"filter-year.value", "filter-airline.value", "filter-month.value", "filter-state.value"}


@app.callback(
    Output("chart-airline", "figure"),
    Output("chart-month",   "figure"),
    Input("filter-year",    "value"),
    Input("filter-airline", "value"),
    Input("filter-month",   "value"),
    Input("filter-state",   "value"),
    State("store-filter-options", "data"),
    prevent_initial_call=True,
)
def update_charts(years_sel, airlines_sel, months_sel, states_sel, options_data):
    if not options_data:
        return _empty_bar("Average delay by airline"), _empty_line("Monthly delay trend")
    if USE_ATHENA:
        clear_athena_error()

    ctx = callback_context
    triggered_ids: set[str] = {t["prop_id"] for t in ctx.triggered} if ctx.triggered else set()

    years_sel    = years_sel    or []
    airlines_sel = airlines_sel or []
    months_sel   = months_sel   or []
    states_sel   = states_sel   or []

    max_years    = int(os.getenv("DASH_MAX_YEARS",    "3"))
    max_airlines = int(os.getenv("DASH_MAX_AIRLINES", "25"))
    max_states   = int(os.getenv("DASH_MAX_STATES",   "25"))

    years_sel    = limit_items(sorted(set(years_sel)),    max_years)
    airlines_sel = limit_items(sorted(set(airlines_sel)), max_airlines)
    states_sel   = limit_items(sorted(set(states_sel)),   max_states)

    only_airline_changed = bool(triggered_ids) and triggered_ids.issubset({"filter-airline.value"})

    if USE_ATHENA:
        airline_fig = query_airline_bar_athena(years_sel, months_sel, airlines_sel, states_sel)
        monthly_fig = (
            no_update
            if only_airline_changed
            else query_monthly_line_athena(years_sel, months_sel, airlines_sel, states_sel)
        )
        return airline_fig, monthly_fig

    df = load_flights_cached(
        years=tuple(sorted(years_sel)),
        months=tuple(sorted(months_sel)),
        airlines=tuple(sorted(airlines_sel)),
        states=tuple(sorted(states_sel)),
    )
    airline_fig = build_airline_bar(df)
    monthly_fig = no_update if only_airline_changed else build_monthly_line(df)
    return airline_fig, monthly_fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
