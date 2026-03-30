import os
import time
from openai import OpenAI
from core.logger import get_logger

logger = get_logger(__name__)


def create_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def chat_completion(client: OpenAI, gpt_version: str, messages: list, **kwargs):
    if "o3-mini" in gpt_version:
        kwargs.setdefault("reasoning_effort", "high")
    return client.chat.completions.create(
        model=gpt_version,
        messages=messages,
        **kwargs,
    )


def chat_completion_raw(client: OpenAI, request_json: dict):
    return client.chat.completions.create(**request_json)


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
