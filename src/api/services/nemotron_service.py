from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
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


@dataclass(frozen=True)
class NemotronAdvice:
    content: str
    provider: str
    model: str


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


def _base_url() -> str:
    raw = os.getenv("NEMOTRON_API_BASE_URL") or os.getenv("NVIDIA_API_BASE_URL") or DEFAULT_BASE_URL
    return raw.strip().rstrip("/")


def _model_name() -> str:
    raw = os.getenv("NEMOTRON_MODEL") or os.getenv("NVIDIA_NEMOTRON_MODEL") or DEFAULT_MODEL
    return raw.strip() or DEFAULT_MODEL


def _api_key(base_url: str) -> str | None:
    raw = (os.getenv("NEMOTRON_API_KEY") or os.getenv("NVIDIA_API_KEY") or "").strip()
    if raw:
        return raw

    parsed = urlparse(base_url)
    host = (parsed.hostname or "").casefold()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return "EMPTY"
    return None


def nemotron_enabled() -> bool:
    if not _env_flag("ADVISOR_LLM_ENABLED", "1"):
        return False
    return _api_key(_base_url()) is not None


def should_use_nemotron(question: str | None) -> bool:
    if not nemotron_enabled():
        return False
    if _env_flag("ADVISOR_LLM_ALWAYS_ON", "0"):
        return True
    return bool((question or "").strip())


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


def _system_prompt() -> str:
    extra = (os.getenv("ADVISOR_LLM_PROMPT_APPEND") or "").strip()
    if not extra:
        return FLIGHT_ADVISOR_SYSTEM_PROMPT
    return f"{FLIGHT_ADVISOR_SYSTEM_PROMPT}\n\n## Regras adicionais\n{extra}"


def _build_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False, indent=2)},
    ]


def generate_nemotron_advice(context: dict[str, Any]) -> NemotronAdvice:
    base_url = _base_url()
    api_key = _api_key(base_url)
    if api_key is None:
        raise RuntimeError("Nemotron API key is not configured.")

    endpoint = f"{base_url}/chat/completions"
    model = _model_name()
    timeout_sec = max(5, _as_int("NEMOTRON_TIMEOUT_SEC", 20))
    max_tokens = max(128, _as_int("NEMOTRON_MAX_TOKENS", 320))

    payload: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(context),
        "max_tokens": max_tokens,
        "temperature": _as_float("NEMOTRON_TEMPERATURE", 1.0),
        "top_p": _as_float("NEMOTRON_TOP_P", 0.95),
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_sec)
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"Nemotron request timed out after {timeout_sec}s.") from exc
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            body = response.json()
            detail = body.get("detail") or body.get("error") or ""
        except ValueError:
            detail = response.text.strip()
        status = response.status_code
        raise RuntimeError(f"Nemotron returned HTTP {status}. {detail}".strip()) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Nemotron connection error: {exc}") from exc

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError("Nemotron returned a non-JSON response.") from exc

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("Nemotron response did not include choices.")

    first = choices[0] or {}
    message = first.get("message") or {}
    content = _flatten_content(message.get("content") or first.get("text"))
    content = _sanitize_output(content)
    if not content:
        raise RuntimeError("Nemotron response was empty.")

    logger.info("Nemotron advice generated with model %s", model)
    return NemotronAdvice(content=content, provider="nvidia_nemotron", model=model)
