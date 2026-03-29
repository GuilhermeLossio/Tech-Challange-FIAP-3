"""
OpenSky Network REST API integration.

Supports anonymous access and OAuth2 client credentials.
OpenSky deprecated username/password basic auth on 2026-03-18.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


def _read_int_env(name: str, default: int, minimum: int = 0) -> int:
    """Read an integer env var with clamping and safe fallback."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s.", name, raw, default)
        return default


def _read_bool_env(name: str, default: bool) -> bool:
    """Read a boolean env var using common truthy tokens."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


OPENSKY_BASE_URL = "https://opensky-network.org/api"
OPENSKY_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/"
    "protocol/openid-connect/token"
)

_TOKEN_REFRESH_MARGIN_SEC = 30
_CACHE_TTL = _read_int_env("OPENSKY_CACHE_TTL_SEC", 15, minimum=1)
_AUTH_TIMEOUT_SEC = _read_int_env("OPENSKY_AUTH_TIMEOUT_SEC", 5, minimum=1)
_TOKEN_RETRY_COOLDOWN_SEC = _read_int_env("OPENSKY_TOKEN_RETRY_COOLDOWN_SEC", 300, minimum=0)
_ALLOW_ANON_FALLBACK = _read_bool_env("OPENSKY_ALLOW_ANON_FALLBACK", True)

# Fields returned by OpenSky for each state vector.
_STATE_FIELDS = [
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "longitude",
    "latitude",
    "baro_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "sensors",
    "geo_altitude",
    "squawk",
    "spi",
    "position_source",
]

_cache: dict[str, Any] = {}
_token_cache: dict[str, Any] = {
    "access_token": None,
    "expires_at": 0.0,
    "retry_after": 0.0,
    "last_error": None,
}
_warned_legacy_basic_auth = False


def _clear_token_backoff() -> None:
    """Clear temporary backoff after a successful authenticated request."""
    _token_cache["retry_after"] = 0.0
    _token_cache["last_error"] = None


def _token_backoff_active(now: float | None = None) -> bool:
    """Return True while auth retries are being temporarily suppressed."""
    current = time.time() if now is None else now
    retry_after = float(_token_cache.get("retry_after") or 0.0)
    return retry_after > current


def _mark_token_backoff(message: str) -> None:
    """Pause OAuth retries for a short window and use anonymous access instead."""
    if _TOKEN_RETRY_COOLDOWN_SEC <= 0:
        return

    retry_after = time.time() + _TOKEN_RETRY_COOLDOWN_SEC
    previous_retry_after = float(_token_cache.get("retry_after") or 0.0)
    _token_cache["access_token"] = None
    _token_cache["expires_at"] = 0.0
    _token_cache["retry_after"] = retry_after
    _token_cache["last_error"] = message

    if retry_after > previous_retry_after:
        logger.warning(
            "OpenSky auth unavailable (%s). Falling back to anonymous access for %ss.",
            message,
            _TOKEN_RETRY_COOLDOWN_SEC,
        )


def _get_client_credentials() -> tuple[str, str] | None:
    """Read optional OAuth2 client credentials from environment variables."""
    client_id = os.getenv("OPENSKY_CLIENT_ID", "").strip()
    client_secret = os.getenv("OPENSKY_CLIENT_SECRET", "").strip()
    return (client_id, client_secret) if client_id and client_secret else None


def _has_legacy_basic_auth() -> bool:
    """Detect deprecated username/password config still present in the env."""
    user = os.getenv("OPENSKY_USERNAME", "").strip()
    password = os.getenv("OPENSKY_PASSWORD", "").strip()
    return bool(user and password)


def _warn_legacy_basic_auth_ignored() -> None:
    """Warn once when deprecated OpenSky basic auth env vars are present."""
    global _warned_legacy_basic_auth
    if _warned_legacy_basic_auth or not _has_legacy_basic_auth():
        return

    logger.warning(
        "Ignoring OPENSKY_USERNAME/OPENSKY_PASSWORD because OpenSky "
        "deprecated basic auth. Configure OPENSKY_CLIENT_ID and "
        "OPENSKY_CLIENT_SECRET to use authenticated requests."
    )
    _warned_legacy_basic_auth = True


def _get_access_token(timeout: int = 10) -> str:
    """Fetch and cache an OpenSky OAuth2 access token."""
    creds = _get_client_credentials()
    if not creds:
        raise RuntimeError(
            "OpenSky OAuth2 credentials are not configured. "
            "Set OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET."
        )

    cached_token = _token_cache.get("access_token")
    expires_at = float(_token_cache.get("expires_at") or 0)
    now = time.time()
    if cached_token and now < expires_at:
        return str(cached_token)

    client_id, client_secret = creds
    try:
        resp = requests.post(
            OPENSKY_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        logger.warning("OpenSky OAuth2 token timeout after %ss", timeout)
        raise TimeoutError("OpenSky authentication timed out. Try again.") from None
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        raise RuntimeError(
            f"OpenSky authentication failed with HTTP {status}. "
            "Check OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"OpenSky authentication error: {exc}") from exc

    payload = resp.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise RuntimeError("OpenSky authentication response did not include access_token.")

    expires_in = int(payload.get("expires_in", 1800))
    _token_cache["access_token"] = access_token
    _token_cache["expires_at"] = now + max(0, expires_in - _TOKEN_REFRESH_MARGIN_SEC)
    _clear_token_backoff()
    return str(access_token)


def _build_headers(timeout: int = 10) -> dict[str, str]:
    """
    Return request headers for OpenSky.

    If OAuth2 credentials are configured, authenticated requests are used.
    Otherwise requests fall back to the anonymous tier.
    """
    creds = _get_client_credentials()
    if not creds:
        _warn_legacy_basic_auth_ignored()
        return {}

    if _ALLOW_ANON_FALLBACK and _token_backoff_active():
        return {}

    auth_timeout = min(timeout, _AUTH_TIMEOUT_SEC)
    try:
        return {"Authorization": f"Bearer {_get_access_token(timeout=auth_timeout)}"}
    except TimeoutError as exc:
        if not _ALLOW_ANON_FALLBACK:
            raise
        _mark_token_backoff(str(exc))
        return {}
    except RuntimeError as exc:
        message = str(exc)
        if not _ALLOW_ANON_FALLBACK or not message.startswith("OpenSky authentication error:"):
            raise
        _mark_token_backoff(message)
        return {}


def _parse_state(raw: list[Any]) -> dict[str, Any]:
    """Convert one OpenSky state vector row into a readable dict."""
    state: dict[str, Any] = {}
    for index, field in enumerate(_STATE_FIELDS):
        state[field] = raw[index] if index < len(raw) else None

    callsign = (state.get("callsign") or "").strip() or None
    state["callsign"] = callsign

    track = state.get("true_track")
    state["heading"] = round(float(track), 1) if track is not None else None

    velocity = state.get("velocity")
    state["speed_kmh"] = round(float(velocity) * 3.6, 1) if velocity is not None else None

    altitude = state.get("baro_altitude")
    state["altitude_ft"] = round(float(altitude) * 3.28084) if altitude is not None else None

    return state


def fetch_live_flights(
    bounding_box: tuple[float, float, float, float] | None = None,
    airport_icao: str | None = None,
    icao24: str | None = None,
    include_ground: bool = False,
    timeout: int = 10,
) -> list[dict[str, Any]]:
    """
    Fetch live in-flight aircraft from OpenSky.

    Parameters
    ----------
    bounding_box:
        (lamin, lomin, lamax, lomax) in decimal degrees.
    airport_icao:
        Optional coarse post-filter by country inferred from ICAO prefix.
    icao24:
        Optional exact ICAO24 filter supported by OpenSky.
    include_ground:
        When True, keep aircraft on the ground as well.
    timeout:
        HTTP timeout in seconds.
    """
    url = f"{OPENSKY_BASE_URL}/states/all"
    params: dict[str, Any] = {}

    if bounding_box:
        lamin, lomin, lamax, lomax = bounding_box
        params.update({"lamin": lamin, "lomin": lomin, "lamax": lamax, "lomax": lomax})

    if icao24:
        params["icao24"] = icao24.strip().lower()

    headers = _build_headers(timeout=timeout)

    try:
        resp = requests.get(url, params=params, headers=headers or None, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        logger.warning("OpenSky timeout after %ss", timeout)
        raise TimeoutError("OpenSky Network did not respond in time. Try again.") from None
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 401:
            raise RuntimeError(
                "OpenSky rejected the request with HTTP 401. "
                "Check OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET."
            ) from exc
        if status == 403:
            raise RuntimeError("OpenSky denied access to this request (HTTP 403).") from exc
        if status == 429:
            raise RuntimeError("OpenSky request limit reached. Wait a few minutes and try again.") from exc
        raise RuntimeError(f"OpenSky returned HTTP {status}: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"OpenSky connection error: {exc}") from exc

    payload = resp.json()
    states = payload.get("states") or []
    timestamp = payload.get("time", int(time.time()))

    flights: list[dict[str, Any]] = []
    for raw in states:
        try:
            state = _parse_state(raw)
        except Exception:
            continue

        if state.get("latitude") is None or state.get("longitude") is None:
            continue

        if state.get("on_ground") and not include_ground:
            continue

        flights.append(
            {
                "icao24": state["icao24"],
                "callsign": state["callsign"],
                "origin_country": state["origin_country"],
                "country": state["origin_country"],
                "latitude": state["latitude"],
                "longitude": state["longitude"],
                "baro_altitude": state["baro_altitude"],
                "altitude_ft": state["altitude_ft"],
                "heading": state["heading"],
                "velocity": state["velocity"],
                "speed_kmh": state["speed_kmh"],
                "on_ground": state["on_ground"],
                "last_contact": state["last_contact"],
                "data_timestamp": timestamp,
            }
        )

    if airport_icao:
        country_map = {
            "SB": "Brazil",
            "GK": "Brazil",
            "EG": "United Kingdom",
            "LF": "France",
            "ED": "Germany",
            "K": "United States",
        }
        prefix = airport_icao[:2].upper()
        country = country_map.get(prefix)
        if country:
            flights = [flight for flight in flights if flight.get("origin_country") == country]

    return flights


def fetch_live_flights_cached(
    bounding_box: tuple[float, float, float, float] | None = None,
    airport_icao: str | None = None,
    icao24: str | None = None,
    include_ground: bool = False,
    ttl: int = _CACHE_TTL,
) -> tuple[list[dict[str, Any]], int]:
    """
    In-memory cached wrapper.

    Returns
    -------
    tuple[list[dict], int]
        (flights, cache_age_seconds)
    """
    key = f"{bounding_box}|{airport_icao}|{icao24}|{include_ground}"
    cached = _cache.get(key)
    if cached:
        age = int(time.time()) - cached["ts"]
        if age < ttl:
            return cached["data"], age

    flights = fetch_live_flights(
        bounding_box=bounding_box,
        airport_icao=airport_icao,
        icao24=icao24,
        include_ground=include_ground,
    )
    _cache[key] = {"data": flights, "ts": int(time.time())}
    return flights, 0


BOUNDING_BOXES = {
    "brazil": (-33.75, -73.99, 5.27, -28.85),
    "south_america": (-56.0, -82.0, 13.0, -32.0),
    "north_america": (14.0, -168.0, 72.0, -52.0),
    "europe": (35.0, -25.0, 72.0, 45.0),
    "world": (-90.0, -180.0, 90.0, 180.0),
}
