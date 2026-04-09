import os
import time
from openai import OpenAI
from core.logger import get_logger

logger = get_logger(__name__)


def _sanitize_string_for_json_payload(text: str) -> str:
    """Normalize potentially invalid unicode before HTTP JSON serialization."""
    if not isinstance(text, str):
        return text
    # Replace unpaired surrogates and other invalid sequences.
    sanitized = text.encode("utf-8", errors="replace").decode("utf-8")
    # Drop NUL bytes that may break some JSON encoders/servers.
    return sanitized.replace("\x00", "")


def _sanitize_payload(obj):
    if isinstance(obj, str):
        return _sanitize_string_for_json_payload(obj)
    if isinstance(obj, list):
        return [_sanitize_payload(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_payload(v) for k, v in obj.items()}
    return obj


def create_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def chat_completion(client: OpenAI, gpt_version: str, messages: list, **kwargs):
    safe_messages = _sanitize_payload(messages)
    safe_kwargs = _sanitize_payload(kwargs)
    if "o3-mini" in gpt_version:
        safe_kwargs.setdefault("reasoning_effort", "high")
    return client.chat.completions.create(
        model=gpt_version,
        messages=safe_messages,
        **safe_kwargs,
    )


def chat_completion_raw(client: OpenAI, request_json: dict):
    safe_request_json = _sanitize_payload(request_json)
    return client.chat.completions.create(**safe_request_json)


_RETRYABLE_STATUS_CODES = (429, 500, 502, 503)


def chat_completion_with_retry(client: OpenAI, gpt_version: str, messages: list,
                               max_retries: int = 3, base_delay: float = 2.0,
                               **kwargs):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return chat_completion(client, gpt_version, messages, **kwargs)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in _RETRYABLE_STATUS_CODES and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"[RETRY] Attempt {attempt+1}/{max_retries} failed ({type(e).__name__}), "
                              f"retrying in {delay:.1f}s...")
                time.sleep(delay)
                last_exc = e
            else:
                raise
    raise last_exc


def chat_completion_raw_with_retry(client: OpenAI, request_json: dict,
                                   max_retries: int = 3, base_delay: float = 2.0):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return chat_completion_raw(client, request_json)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in _RETRYABLE_STATUS_CODES and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"[RETRY] Attempt {attempt+1}/{max_retries} failed ({type(e).__name__}), "
                               f"retrying in {delay:.1f}s...")
                time.sleep(delay)
                last_exc = e
            else:
                raise
    raise last_exc
