"""Groq-specific LiteLLM wrapper with TPM-aware throttling, 429 retry, and model fallback."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

from smolagents.models import (
    ChatMessage,
    LiteLLMModel,
    Model,
    TokenUsage,
    is_rate_limit_error,
    remove_content_after_stop_sequences,
)

GROQ_RETRY_AFTER_PATTERN = re.compile(r"try again in ([\d.]+)s", re.I)
GROQ_RETRY_AFTER_HMS_PATTERN = re.compile(
    r"try again in (?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>[\d.]+)s)?",
    re.I,
)
GROQ_DEFAULT_SPACE_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_DEFAULT_LOCAL_MODEL = "llama-3.3-70b-versatile"

# Large-context free-tier chat models only (multi-step agent traces exceed small windows).
DEFAULT_GROQ_FALLBACK_CHAIN = (
    "qwen/qwen3-32b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
    "qwen/qwen3.6-27b",
    "llama-3.3-70b-versatile",
)

QUOTA_EXHAUSTED_PATTERNS = (
    "tokens per day",
    "requests per day",
    '"type":"tokens"',
    '"type": "tokens"',
)

UNAVAILABLE_MODEL_PATTERNS = (
    "model_not_found",
    "does not exist",
    "do not have access",
)

CONTEXT_LIMIT_PATTERNS = (
    "context_length_exceeded",
    "context window",
    "maximum context",
    "too many tokens",
    "reduce the length",
)

T = TypeVar("T")


def groq_min_request_interval() -> float:
    raw = os.getenv("GROQ_MIN_REQUEST_INTERVAL", "5")
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 3.0


def groq_requests_per_minute() -> float | None:
    interval = groq_min_request_interval()
    if interval <= 0:
        return None
    return 60.0 / interval


def groq_max_retries() -> int:
    raw = os.getenv("GROQ_MAX_RETRIES", "5")
    try:
        return max(int(raw), 1)
    except ValueError:
        return 5


def groq_fallback_enabled() -> bool:
    raw = os.getenv("GROQ_FALLBACK_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def groq_max_retry_wait() -> float:
    raw = os.getenv("GROQ_MAX_RETRY_WAIT", "120")
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 120.0


def parse_groq_retry_seconds(error_message: str) -> float | None:
    hms_match = GROQ_RETRY_AFTER_HMS_PATTERN.search(error_message)
    if hms_match:
        hours = int(hms_match.group("hours") or 0)
        minutes = int(hms_match.group("minutes") or 0)
        seconds = float(hms_match.group("seconds") or 0)
        total = hours * 3600 + minutes * 60 + seconds
        if total > 0:
            return total + 1.0

    match = GROQ_RETRY_AFTER_PATTERN.search(error_message)
    if not match:
        return None
    return float(match.group(1)) + 1.0


def is_groq_quota_exhausted_error(error: BaseException) -> bool:
    """Daily quota (TPD/RPD) — do not retry; switch model immediately."""
    message = str(error).lower()
    if any(pattern in message for pattern in QUOTA_EXHAUSTED_PATTERNS):
        return True
    if "rate limit reached" in message and "per day" in message:
        return True
    wait = parse_groq_retry_seconds(str(error))
    return wait is not None and wait > groq_max_retry_wait()


def groq_retry_wait_seconds(error: BaseException, attempt: int) -> float:
    parsed = parse_groq_retry_seconds(str(error))
    if parsed is not None:
        return parsed
    base = float(os.getenv("GROQ_RETRY_WAIT", "15"))
    return base * attempt


def is_groq_unavailable_model_error(error: BaseException) -> bool:
    """Model id invalid or inaccessible — skip to next model immediately."""
    message = str(error).lower()
    if "404" in message and "model" in message:
        return True
    return any(pattern in message for pattern in UNAVAILABLE_MODEL_PATTERNS)


def is_groq_context_limit_error(error: BaseException) -> bool:
    """Prompt too long for this model — try a larger-context fallback."""
    message = str(error).lower()
    return any(pattern in message for pattern in CONTEXT_LIMIT_PATTERNS)


def is_groq_rate_limit_error(error: BaseException) -> bool:
    """TPM/RPM style limits including Groq 413 responses."""
    if is_rate_limit_error(error):
        return True
    message = str(error).lower()
    return any(
        pattern in message
        for pattern in ("rate_limit_exceeded", "rate limit reached", "too many requests", "413", "429")
    )


def is_groq_soft_tpm_error(error: BaseException) -> bool:
    """Brief TPM burst — worth retrying on the same model."""
    if (
        is_groq_quota_exhausted_error(error)
        or is_groq_context_limit_error(error)
        or is_groq_unavailable_model_error(error)
    ):
        return False
    message = str(error).lower()
    if "413" in message:
        return False
    if not is_groq_rate_limit_error(error):
        return False
    wait = parse_groq_retry_seconds(str(error))
    if wait is not None:
        return wait <= groq_max_retry_wait()
    return "429" in message and "rate_limit_exceeded" not in message


def is_groq_fallback_error(error: BaseException) -> bool:
    """Switch to the next model in the chain (do not retry the same model)."""
    return (
        is_groq_unavailable_model_error(error)
        or is_groq_context_limit_error(error)
        or is_hard_groq_limit_error(error)
    )


def is_hard_groq_limit_error(error: BaseException) -> bool:
    """True when the current model cannot serve this request (switch to fallback)."""
    if is_groq_quota_exhausted_error(error):
        return True
    return is_groq_rate_limit_error(error)


def _groq_failure_reason(error: BaseException) -> str:
    if is_groq_unavailable_model_error(error):
        return "unavailable"
    if is_groq_context_limit_error(error):
        return "context too long"
    return "limit hit"


def build_groq_model_chain(normalize: Callable[[str], str]) -> list[str]:
    primary = normalize(default_groq_model_id())
    chain_env = os.getenv("GROQ_MODEL_FALLBACK_CHAIN", "").strip()
    if chain_env:
        fallbacks = [normalize(part.strip()) for part in chain_env.split(",") if part.strip()]
    else:
        fallbacks = [normalize(model_id) for model_id in DEFAULT_GROQ_FALLBACK_CHAIN]

    chain = [primary]
    for model_id in fallbacks:
        if model_id not in chain:
            chain.append(model_id)
    return chain


def call_with_groq_retry(fn: Callable[..., T], **kwargs: Any) -> T:
    max_attempts = groq_max_retries()
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(**kwargs)
        except Exception as error:
            last_error = error
            if not is_groq_soft_tpm_error(error):
                raise
            if attempt >= max_attempts:
                raise
            wait = groq_retry_wait_seconds(error, attempt)
            if wait > groq_max_retry_wait():
                raise
            print(
                f"Groq rate limit (attempt {attempt}/{max_attempts}), "
                f"waiting {wait:.1f}s..."
            )
            time.sleep(wait)
    if last_error:
        raise last_error
    raise RuntimeError("Groq retry loop exited without result")


def default_groq_model_id() -> str:
    configured = os.getenv("GROQ_MODEL", "").strip()
    if configured:
        return configured
    if os.getenv("SPACE_ID"):
        return GROQ_DEFAULT_SPACE_MODEL
    return GROQ_DEFAULT_LOCAL_MODEL


class GroqLiteLLMModel(LiteLLMModel):
    """LiteLLM model with Groq TPM throttling and parsed 429 backoff."""

    def __init__(self, *, quiet: bool = False, **kwargs: Any) -> None:
        rpm = groq_requests_per_minute()
        kwargs.setdefault("requests_per_minute", rpm)
        kwargs.setdefault("retry", False)
        super().__init__(**kwargs)
        if quiet:
            return
        interval = groq_min_request_interval()
        rpm_label = f"{rpm:.1f}" if rpm is not None else "off"
        print(
            f"Groq throttling: min interval={interval}s ({rpm_label} RPM), "
            f"max retries={groq_max_retries()}"
        )

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )
        self._apply_rate_limit()
        response = call_with_groq_retry(self.client.completion, **completion_kwargs)

        if not response.choices:
            raise RuntimeError(
                f"Unexpected API response: model '{self.model_id}' returned no choices. "
                "This may indicate a possible API or upstream issue. "
                f"Response details: {response.model_dump()}"
            )
        content = response.choices[0].message.content
        if stop_sequences is not None and not self.supports_stop_parameter:
            content = remove_content_after_stop_sequences(content, stop_sequences)
        return ChatMessage(
            role=response.choices[0].message.role,
            content=content,
            tool_calls=response.choices[0].message.tool_calls,
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

