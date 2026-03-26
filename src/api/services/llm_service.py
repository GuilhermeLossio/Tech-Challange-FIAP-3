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
DEFAULT_TEMPERATURE = 0.4
# Janela máxima de mensagens enviadas ao LLM (evita estouro de contexto)
MAX_HISTORY_MESSAGES = 10
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 1.5
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

FLIGHT_ADVISOR_SYSTEM_PROMPT = """
## Papel
Você é um Flight Advisor, assistente especializado em viagens aéreas. Você atende viajantes corporativos, turistas, famílias e viajantes solo. Seu objetivo é ajudar o cliente a encontrar, comparar e comprar passagens aéreas da melhor forma possível.

## Comportamento principal
1. Seja proativo. Não bloqueie a conversa esperando todos os dados. Se faltar data, sugira datas próximas com bom custo-benefício. Se faltar origem, pergunte de forma natural ou use o contexto disponível. Se faltar número de passageiros, assuma 1 adulto e informe isso ao cliente. Se o destino estiver em aberto, sugira opções com base no perfil e nos interesses citados.
2. Nunca invente voos, preços, horários, disponibilidade, aeroportos, probabilidades ou confirmações de compra.
3. Use apenas os dados estruturados disponíveis na mensagem do usuário e respeite estritamente o objeto `assistant_runtime` enviado nela.
4. Se `assistant_runtime.tooling.search_flights.enabled` for true e houver resultados estruturados de busca, apresente opções reais de forma clara e comparativa. Se estiver false, deixe claro que a busca em tempo real ainda não está integrada neste backend e não finja ter executado a consulta.
5. Faça sugestões inteligentes com base em custo-benefício, preferências inferidas e perfil do cliente. Explique brevemente o motivo de cada sugestão.
6. Só conduza fluxo de compra se `assistant_runtime.tooling.booking_flow.enabled` for true. Nunca finalize compra sem confirmação explícita do cliente.
7. Se a API ou a ferramenta não tiver resultados, informe isso com transparência e proponha alternativas, como datas próximas, aeroportos alternativos ou ajustes de rota.
8. Não armazene dados sensíveis além da sessão atual.

## Tom e comunicação
- Seja direto, amigável e eficiente.
- Adapte o vocabulário ao perfil do cliente.
- Evite jargões de aviação sem explicar.
- Em caso de dúvida sobre o perfil, use tom neutro e profissional.
- Responda em português do Brasil, a menos que o cliente tenha escrito claramente em outro idioma.
""".strip()

DISCOVERY_STYLE_PROMPT = """
## Discovery
- Se a pergunta for ampla, exploratória ou focada em planejamento, não pule direto para atraso.
- Primeiro entenda o que o cliente realmente precisa.
- Nesses casos, responda de forma mais amigável, consultiva e acolhedora.
- Se houver destino ou região no contexto, você pode comentar clima típico, perfil da região, melhor época em termos gerais e atividades turísticas comuns.
- Quando falar de clima ou weather, trate isso como orientação geral da região, nunca como previsão em tempo real, a menos que exista uma ferramenta ou dado específico para isso.
- Só mencione probabilidade, risco, fatores de atraso ou números operacionais quando o cliente pedir isso de forma explícita.
""".strip()


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
# Helpers de configuração via variáveis de ambiente
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
# Processamento de mensagens
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


def _system_prompt(system_prompt: str | None = None) -> str:
    base_prompt = (system_prompt or FLIGHT_ADVISOR_SYSTEM_PROMPT).strip()
    extra = (os.getenv("ADVISOR_LLM_PROMPT_APPEND") or "").strip()
    if not extra:
        return f"{base_prompt}\n\n{DISCOVERY_STYLE_PROMPT}"
    return f"{base_prompt}\n\n{DISCOVERY_STYLE_PROMPT}\n\n## Regras adicionais\n{extra}"


def _trim_history(messages: list[dict[str, Any]], max_messages: int = MAX_HISTORY_MESSAGES) -> list[dict[str, Any]]:
    non_system = [m for m in messages if m.get("role") != "system"]
    if len(non_system) > max_messages:
        logger.debug(
            "Histórico truncado de %d para %d mensagens antes de enviar ao LLM.",
            len(non_system),
            max_messages,
        )
        non_system = non_system[-max_messages:]
    return non_system


def _build_messages(
    context: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """
    Monta o array de mensagens para a API.
    - Inclui o system prompt
    - Injeta histórico recente (já truncado)
    """
    msgs: list[dict[str, str]] = [{"role": "system", "content": _system_prompt(system_prompt)}]

    if history:
        trimmed = _trim_history(history)
        msgs.extend(trimmed)

    # Contexto atual serializado de forma compacta
    msgs.append({
        "role": "user",
        "content": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
    })
    return msgs


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def _iter_stream_chunks(response: requests.Response) -> Generator[str, None, None]:
    """Itera sobre chunks SSE e extrai os deltas de texto."""
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
    max_tokens = max(128, _as_int("NEMOTRON_MAX_TOKENS", DEFAULT_MAX_TOKENS))

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": _build_messages(context, history, system_prompt),
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
# Chamada síncrona principal (com retry)
# ---------------------------------------------------------------------------

def generate_llm_advice(
    context: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
) -> LLMAdvice:
    config = _provider_config()
    endpoint = f"{config.base_url}/chat/completions"
    timeout_sec = max(5, _as_int("NEMOTRON_TIMEOUT_SEC", 20))
    max_tokens = max(128, _as_int("NEMOTRON_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    attempts = max(1, _as_int("NEMOTRON_RETRY_ATTEMPTS", RETRY_ATTEMPTS))

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": _build_messages(context, history, system_prompt),
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
                logger.warning("Timeout na tentativa %d/%d. Aguardando %.1fs...", attempt, attempts, wait)
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
                logger.warning("Erro de conexão na tentativa %d/%d: %s", attempt, attempts, exc)
                time.sleep(wait)
                continue
            raise RuntimeError(f"{config.label} connection error: {exc}") from exc

        # Sucesso — processa resposta
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
