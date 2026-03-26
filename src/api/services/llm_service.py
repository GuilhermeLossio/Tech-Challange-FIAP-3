from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Generator
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"
DEFAULT_HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_PROVIDER = "nvidia"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_COMPACT_MAX_TOKENS = 512
DEFAULT_GUIDE_MAX_TOKENS = 2800
DEFAULT_TEMPERATURE = 0.4
# Maximum number of messages sent to the LLM to avoid context overflow
MAX_HISTORY_MESSAGES = 10
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 1.5
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

FLIGHT_ADVISOR_SYSTEM_PROMPT = """
## Role
You are Flight Advisor, a specialist assistant for air travel. You help business travelers, tourists, families, and solo travelers. Your goal is to help the customer find, compare, and buy airline tickets in the best possible way.

## Core behavior
1. Be proactive. Do not block the conversation waiting for every detail. If the date is missing, suggest nearby dates with good value. If the origin is missing, ask naturally or use the available context. If the number of passengers is missing, assume 1 adult and tell the customer. If the destination is still open, suggest options based on the profile and interests mentioned.
2. Never invent flights, prices, schedules, availability, airports, probabilities, or purchase confirmations.
3. Use only the structured data available in the user's message and strictly follow the `assistant_runtime` object included in it.
4. If `assistant_runtime.tooling.search_flights.enabled` is true and there are structured search results, present real options clearly and comparatively. If it is false, make it clear that real-time search is not integrated in this backend yet and do not pretend the query was executed.
5. Make smart suggestions based on value, inferred preferences, and the customer's profile. Briefly explain the reason behind each suggestion.
6. Only move into a booking flow if `assistant_runtime.tooling.booking_flow.enabled` is true. Never complete a purchase without the customer's explicit confirmation.
7. If the API or tool has no results, say so transparently and propose alternatives such as nearby dates, alternate airports, or route adjustments.
8. Do not store sensitive data beyond the current session.

## Tone and communication
- Be direct, friendly, and efficient.
- Adapt the vocabulary to the customer's profile.
- Avoid aviation jargon unless you explain it.
- If the user's profile is unclear, use a neutral and professional tone.
- Reply in Brazilian Portuguese unless the customer clearly wrote in another language.
""".strip()

DISCOVERY_STYLE_PROMPT = """
## Discovery
- If the question is broad, exploratory, or focused on planning, do not jump straight to delays.
- First understand what the customer actually needs.
- In those cases, respond in a friendlier, more consultative, and more welcoming way.
- If there is a destination or region in the context, you may comment on the typical climate, the general profile of the region, the usual best time to visit, and common tourist activities.
- When talking about climate or weather, treat it as general regional guidance, never as a real-time forecast unless there is a specific tool or dataset for that.
- Only mention probability, risk, delay factors, or operational numbers when the customer asks for that explicitly.
""".strip()

COMPACT_SYSTEM_PROMPT = """
You are a concise flight advisor.
- Reply in the user's language. Default to Brazilian Portuguese.
- Use only the structured context provided.
- Never invent flights, prices, availability, delay metrics, or live weather.
- If the request is broad, stay in discovery mode and avoid delay metrics unless explicitly asked.
- If key data is missing, ask only one or two short follow-up questions.
- Be direct, practical, and brief.
""".strip()


def _question_from_context(context: dict[str, Any] | None) -> str:
    if not isinstance(context, dict):
        return ""
    question = context.get("question")
    if isinstance(question, str):
        return question.strip()
    return ""


def _is_complete_travel_guide_request(context: dict[str, Any] | None) -> bool:
    text = _question_from_context(context).casefold()
    if not text:
        return False

    explicit_markers = (
        "complete travel guide",
        "full travel guide",
        "travel guide from",
        "guia completo",
        "roteiro completo",
        "do not leave any section incomplete",
        "respond in english",
        "best time to visit",
        "top tourist attractions",
        "conclusion with next steps",
    )
    if any(marker in text for marker in explicit_markers):
        return True

    guide_terms = (
        "guide", "travel guide", "guia", "roteiro",
    )
    content_terms = (
        "climate", "weather", "attractions", "tourist attractions",
        "gastronomy", "food", "accommodation", "hotel", "transportation",
        "gastronomia", "hospedagem", "transporte", "clima",
    )
    return any(term in text for term in guide_terms) and any(term in text for term in content_terms)


@dataclass(frozen=True)
class LLMAdvice:
    content: str
    provider: str
    model: str


NemotronAdvice = LLMAdvice


@dataclass(frozen=True)
class LLMProviderConfig:
    provider: str
    label: str
    base_url: str
    api_key: str
    model: str


# ---------------------------------------------------------------------------
# Environment-variable configuration helpers
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _as_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _provider_name() -> str:
    raw = (os.getenv("ADVISOR_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().casefold()
    if raw in {"hf", "huggingface", "hugging_face", "huggingface_router", "hf_router"}:
        return "huggingface"
    return "nvidia"


def _provider_label(provider: str) -> str:
    if provider == "huggingface":
        return "huggingface"
    return "nvidia"


def _base_url(provider: str | None = None) -> str:
    provider = provider or _provider_name()
    if provider == "huggingface":
        raw = os.getenv("HUGGINGFACE_API_BASE_URL") or os.getenv("HF_API_BASE_URL") or DEFAULT_HUGGINGFACE_BASE_URL
    else:
        raw = os.getenv("NEMOTRON_API_BASE_URL") or os.getenv("NVIDIA_API_BASE_URL") or DEFAULT_BASE_URL
    return raw.strip().rstrip("/")


def _shared_model_name() -> str | None:
    raw = (os.getenv("ADVISOR_LLM_MODEL") or os.getenv("LLM_MODEL") or "").strip()
    return raw or None


def _normalize_model_name(provider: str, model: str) -> str:
    normalized = (model or "").strip()
    if not normalized:
        return normalized
    if provider == "huggingface":
        prefixes = (
            "https://huggingface.co/",
            "http://huggingface.co/",
            "https://www.huggingface.co/",
            "http://www.huggingface.co/",
            "huggingface.co/",
            "//huggingface.co/",
        )
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        normalized = normalized.strip().strip("/")
    return normalized


def _model_name(provider: str | None = None) -> str:
    provider = provider or _provider_name()
    shared = _shared_model_name()
    if shared:
        return _normalize_model_name(provider, shared)
    if provider == "huggingface":
        raw = os.getenv("HUGGINGFACE_MODEL") or os.getenv("HF_MODEL") or ""
        return _normalize_model_name(provider, raw)
    raw = os.getenv("NEMOTRON_MODEL") or os.getenv("NVIDIA_NEMOTRON_MODEL") or DEFAULT_MODEL
    normalized = _normalize_model_name(provider, raw)
    return normalized or DEFAULT_MODEL


def _api_key(provider: str, base_url: str) -> str | None:
    if provider == "huggingface":
        raw = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_API_KEY")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
            or ""
        ).strip()
    else:
        raw = (os.getenv("NEMOTRON_API_KEY") or os.getenv("NVIDIA_API_KEY") or "").strip()
    if raw:
        return raw
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").casefold()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return "EMPTY"
    return None


def _provider_config() -> LLMProviderConfig:
    provider = _provider_name()
    base_url = _base_url(provider)
    api_key = _api_key(provider, base_url)
    if api_key is None:
        if provider == "huggingface":
            raise RuntimeError("Hugging Face API key is not configured.")
        raise RuntimeError("NVIDIA API key is not configured.")

    model = _model_name(provider)
    if not model:
        if provider == "huggingface":
            raise RuntimeError("Hugging Face model is not configured.")
        raise RuntimeError("NVIDIA model is not configured.")

    return LLMProviderConfig(
        provider=provider,
        label=_provider_label(provider),
        base_url=base_url,
        api_key=api_key,
        model=model,
    )


def _compact_mode_enabled(config: LLMProviderConfig, context: dict[str, Any] | None = None) -> bool:
    if _is_complete_travel_guide_request(context):
        return False
    if _env_flag("ADVISOR_LLM_COMPACT_MODE", "0"):
        return True
    return "qwen" in (config.model or "").casefold()


def llm_enabled() -> bool:
    if not _env_flag("ADVISOR_LLM_ENABLED", "1"):
        return False
    provider = _provider_name()
    return _api_key(provider, _base_url(provider)) is not None and bool(_model_name(provider))


def should_use_llm(question: str | None) -> bool:
    if not llm_enabled():
        return False
    if _env_flag("ADVISOR_LLM_ALWAYS_ON", "0"):
        return True
    return bool((question or "").strip())


nemotron_enabled = llm_enabled
should_use_nemotron = should_use_llm


# ---------------------------------------------------------------------------
# Message processing
# ---------------------------------------------------------------------------

def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(part.strip() for part in parts if str(part).strip())
    return str(content or "")


def _sanitize_output(text: str) -> str:
    cleaned = text.strip()
    if "<think>" in cleaned and "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    return cleaned


def _system_prompt(system_prompt: str | None = None, compact: bool = False) -> str:
    extra = (os.getenv("ADVISOR_LLM_PROMPT_APPEND") or "").strip()
    if compact:
        return f"{COMPACT_SYSTEM_PROMPT}\n\n## Additional rules\n{extra}" if extra else COMPACT_SYSTEM_PROMPT

    base_prompt = (system_prompt or FLIGHT_ADVISOR_SYSTEM_PROMPT).strip()
    if not extra:
        return f"{base_prompt}\n\n{DISCOVERY_STYLE_PROMPT}"
    return f"{base_prompt}\n\n{DISCOVERY_STYLE_PROMPT}\n\n## Additional rules\n{extra}"


def _history_limit(config: LLMProviderConfig, context: dict[str, Any] | None = None) -> int:
    if _is_complete_travel_guide_request(context):
        return max(2, _as_int("ADVISOR_LLM_GUIDE_MAX_HISTORY_MESSAGES", 6))
    if _compact_mode_enabled(config, context):
        return max(2, _as_int("QWEN_MAX_HISTORY_MESSAGES", 4))
    return max(2, _as_int("ADVISOR_LLM_MAX_HISTORY_MESSAGES", MAX_HISTORY_MESSAGES))


def _trim_history(messages: list[dict[str, Any]], max_messages: int = MAX_HISTORY_MESSAGES) -> list[dict[str, Any]]:
    non_system = [m for m in messages if m.get("role") != "system"]
    if len(non_system) > max_messages:
        logger.debug(
            "History trimmed from %d to %d messages before sending to the LLM.",
            len(non_system),
            max_messages,
        )
        non_system = non_system[-max_messages:]
    return non_system


def _prune_context(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            pruned = _prune_context(item)
            if pruned in (None, "", [], {}):
                continue
            cleaned[key] = pruned
        return cleaned
    if isinstance(value, list):
        cleaned_items = [_prune_context(item) for item in value]
        return [item for item in cleaned_items if item not in (None, "", [], {})]
    return value


def _compact_context(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        if key == "assistant_runtime":
            tooling = value.get("tooling") or {}
            restrictions = value.get("restrictions") or {}
            compact_runtime = {
                "tooling": {
                    "search_flights_enabled": bool((tooling.get("search_flights") or {}).get("enabled")),
                    "booking_flow_enabled": bool((tooling.get("booking_flow") or {}).get("enabled")),
                    "fare_history_enabled": bool((tooling.get("fare_history") or {}).get("enabled")),
                },
                "restrictions": restrictions,
            }
            return _prune_context(compact_runtime)

        compact_dict: dict[str, Any] = {}
        for child_key, child_value in value.items():
            compact_value = _compact_context(child_value, child_key)
            if compact_value in (None, "", [], {}):
                continue
            compact_dict[child_key] = compact_value
        return compact_dict

    if isinstance(value, list):
        item_limit = {
            "top_factors": 3,
            "suggested_flights": 2,
            "clarification_prompts": 2,
            "allowed_topics": 3,
            "missing_fields": 4,
        }.get(key, 4)
        compact_list = []
        for item in value[:item_limit]:
            compact_value = _compact_context(item)
            if compact_value in (None, "", [], {}):
                continue
            compact_list.append(compact_value)
        return compact_list

    if isinstance(value, str):
        text = value.strip()
        if len(text) > 280:
            return text[:277].rstrip() + "..."
        return text

    return value


def _build_messages(
    config: LLMProviderConfig,
    context: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """
    Build the message array for the API.
    - Includes the system prompt
    - Injects the recent history after trimming
    """
    compact_mode = _compact_mode_enabled(config, context)
    msgs: list[dict[str, str]] = [{"role": "system", "content": _system_prompt(system_prompt, compact=compact_mode)}]

    if history:
        trimmed = _trim_history(history, max_messages=_history_limit(config, context))
        msgs.extend(trimmed)

    context_payload = _prune_context(_compact_context(context) if compact_mode else context)
    msgs.append({
        "role": "user",
        "content": json.dumps(context_payload, ensure_ascii=False, separators=(",", ":")),
    })
    return msgs


def _resolve_max_tokens(config: LLMProviderConfig, context: dict[str, Any] | None = None) -> int:
    if _is_complete_travel_guide_request(context):
        return max(512, _as_int("ADVISOR_LLM_GUIDE_MAX_TOKENS", DEFAULT_GUIDE_MAX_TOKENS))
    if _compact_mode_enabled(config, context):
        return max(96, _as_int("QWEN_MAX_TOKENS", DEFAULT_COMPACT_MAX_TOKENS))
    return max(128, _as_int("NEMOTRON_MAX_TOKENS", DEFAULT_MAX_TOKENS))


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def _iter_stream_chunks(response: requests.Response) -> Generator[str, None, None]:
    """Iterate over SSE chunks and extract text deltas."""
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
        piece = delta.get("content") or ""
        if piece:
            yield piece


def generate_llm_advice_stream(
    context: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
) -> Generator[str, None, None]:
    config = _provider_config()
    endpoint = f"{config.base_url}/chat/completions"
    timeout_sec = max(5, _as_int("NEMOTRON_TIMEOUT_SEC", 30))
    max_tokens = _resolve_max_tokens(config, context)

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": _build_messages(config, context, history, system_prompt),
        "max_tokens": max_tokens,
        "temperature": _as_float("NEMOTRON_TEMPERATURE", DEFAULT_TEMPERATURE),
        "top_p": _as_float("NEMOTRON_TOP_P", 0.95),
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    with requests.post(endpoint, headers=headers, json=payload, timeout=timeout_sec, stream=True) as resp:
        resp.raise_for_status()
        yield from _iter_stream_chunks(resp)


# ---------------------------------------------------------------------------
# Main synchronous call with retry
# ---------------------------------------------------------------------------

def generate_llm_advice(
    context: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
) -> LLMAdvice:
    config = _provider_config()
    endpoint = f"{config.base_url}/chat/completions"
    timeout_sec = max(5, _as_int("NEMOTRON_TIMEOUT_SEC", 20))
    max_tokens = _resolve_max_tokens(config, context)
    attempts = max(1, _as_int("NEMOTRON_RETRY_ATTEMPTS", RETRY_ATTEMPTS))

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": _build_messages(config, context, history, system_prompt),
        "max_tokens": max_tokens,
        "temperature": _as_float("NEMOTRON_TEMPERATURE", DEFAULT_TEMPERATURE),
        "top_p": _as_float("NEMOTRON_TOP_P", 0.95),
    }

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                endpoint, headers=headers, json=payload, timeout=timeout_sec
            )
            if response.status_code in RETRY_STATUS_CODES and attempt < attempts:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "LLM provider %s returned HTTP %d (attempt %d/%d). Waiting %.1fs...",
                    config.label,
                    response.status_code, attempt, attempts, wait,
                )
                time.sleep(wait)
                continue

            response.raise_for_status()

        except requests.exceptions.Timeout as exc:
            last_exc = exc
            if attempt < attempts:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning("Timeout on attempt %d/%d. Waiting %.1fs...", attempt, attempts, wait)
                time.sleep(wait)
                continue
            raise RuntimeError(f"{config.label} request timed out after {timeout_sec}s.") from exc

        except requests.exceptions.HTTPError as exc:
            detail = ""
            try:
                body = response.json()
                detail = body.get("detail") or body.get("error") or ""
            except ValueError:
                detail = response.text.strip()
            status = response.status_code
            raise RuntimeError(f"{config.label} returned HTTP {status}. {detail}".strip()) from exc

        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < attempts:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning("Connection error on attempt %d/%d: %s", attempt, attempts, exc)
                time.sleep(wait)
                continue
            raise RuntimeError(f"{config.label} connection error: {exc}") from exc

        # Success: process response
        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(f"{config.label} returned a non-JSON response.") from exc

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"{config.label} response did not include choices.")

        first = choices[0] or {}
        message = first.get("message") or {}
        content = _flatten_content(message.get("content") or first.get("text"))
        content = _sanitize_output(content)
        if not content:
            raise RuntimeError(f"{config.label} response was empty.")

        logger.info("%s advice generated with model %s (attempt %d/%d)", config.label, config.model, attempt, attempts)
        return LLMAdvice(content=content, provider=config.label, model=config.model)

    raise RuntimeError(f"{config.label} failed after {attempts} attempts.") from last_exc


generate_nemotron_advice_stream = generate_llm_advice_stream
generate_nemotron_advice = generate_llm_advice

__all__ = [
    "LLMAdvice",
    "LLMProviderConfig",
    "llm_enabled",
    "should_use_llm",
    "generate_llm_advice",
    "generate_llm_advice_stream",
    "NemotronAdvice",
    "nemotron_enabled",
    "should_use_nemotron",
    "generate_nemotron_advice",
    "generate_nemotron_advice_stream",
]
