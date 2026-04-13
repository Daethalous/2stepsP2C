import json
import os
import re
import time
from typing import Any
from openai import OpenAI
from core.logger import get_logger

logger = get_logger(__name__)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_string_for_json_payload(text: str) -> str:
    """Normalize potentially invalid unicode before HTTP JSON serialization."""
    if not isinstance(text, str):
        return text
    # Replace unpaired surrogates and other invalid sequences.
    sanitized = text.encode("utf-8", errors="replace").decode("utf-8")
    # Drop control characters that may break JSON encoders/servers.
    return CONTROL_CHAR_RE.sub(" ", sanitized.replace("\x00", ""))


def sanitize_prompt_text(text: Any, max_chars: int = None) -> str:
    if text is None:
        sanitized = ""
    elif isinstance(text, str):
        sanitized = text
    else:
        sanitized = str(text)
    sanitized = _sanitize_string_for_json_payload(sanitized)
    if max_chars is not None and len(sanitized) > max_chars:
        sanitized = sanitized[:max_chars] + "\n...(truncated for token budget)..."
    return sanitized


def sanitize_payload(obj):
    if isinstance(obj, str):
        return _sanitize_string_for_json_payload(obj)
    if isinstance(obj, list):
        return [sanitize_payload(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_payload(v) for k, v in obj.items()}
    return obj

def sanitize_and_dump_reload(obj):
    # Ensure what pydantic serializes is exactly what JSON handles cleanly natively
    safe_obj = sanitize_payload(obj)
    return json.loads(json.dumps(safe_obj, ensure_ascii=False))

def prepare_messages_for_api(messages: list, **kwargs) -> tuple[list, dict, int]:
    safe_messages = sanitize_and_dump_reload(messages)
    safe_kwargs = sanitize_and_dump_reload(kwargs)
    dumped = json.dumps(
        {
            "model": safe_kwargs.get("model", ""),
            "messages": safe_messages,
            **{k: v for k, v in safe_kwargs.items() if k != "model"},
        },
        ensure_ascii=False,
    )
    dumped.encode("utf-8", errors="strict")
    return safe_messages, safe_kwargs, len(dumped)


def create_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def chat_completion(client: OpenAI, gpt_version: str, messages: list, **kwargs):
    safe_messages = sanitize_and_dump_reload(messages)
    safe_kwargs = sanitize_and_dump_reload(kwargs)
    if "o3-mini" in gpt_version:
        safe_kwargs.setdefault("reasoning_effort", "high")
    return client.chat.completions.create(
        model=gpt_version,
        messages=safe_messages,
        **safe_kwargs,
    )


def chat_completion_raw(client: OpenAI, request_json: dict):
    safe_request_json = sanitize_payload(request_json)
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
