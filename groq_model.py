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

# Standard Groq chat models (CodeAgent-compatible). Do not use groq/compound* here —
# LiteLLM sends the wrong id and compound models reject custom client tools.
DEFAULT_GROQ_FALLBACK_CHAIN = (
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
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

T = TypeVar("T")


def groq_min_request_interval() -> float:
    raw = os.getenv("GROQ_MIN_REQUEST_INTERVAL", "3")
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


def is_groq_fallback_error(error: BaseException) -> bool:
    """Switch to the next model in the chain (do not retry the same model)."""
    return is_groq_unavailable_model_error(error) or is_hard_groq_limit_error(error)


def is_hard_groq_limit_error(error: BaseException) -> bool:
    """True when the current model cannot serve this request (switch to fallback)."""
    if is_groq_quota_exhausted_error(error):
        return True
    if is_rate_limit_error(error):
        return True
    message = str(error).lower()
    hard_patterns = (
        "rate_limit_exceeded",
        "rate limit reached",
        "too many requests",
        "429",
    )
    return any(pattern in message for pattern in hard_patterns)


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
            if is_groq_quota_exhausted_error(error) or is_groq_unavailable_model_error(error):
                raise
            if not is_rate_limit_error(error) or attempt >= max_attempts:
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


class GroqFallbackModel(Model):
    """Try Groq models in order; switch when a model hits a hard rate or quota limit."""

    def __init__(
        self,
        model_ids: list[str],
        api_key: str,
        api_base: str,
        temperature: float = 0,
    ) -> None:
        if not model_ids:
            raise ValueError("Groq model chain cannot be empty")
        super().__init__(model_id=model_ids[0], temperature=temperature)
        self.model_ids = model_ids
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self._models: dict[str, GroqLiteLLMModel] = {}
        self._active_index = 0
        self._exhausted: set[str] = set()
        print(f"Groq fallback chain: {' -> '.join(model_ids)}")

    @property
    def active_model_id(self) -> str:
        for index in range(self._active_index, len(self.model_ids)):
            model_id = self.model_ids[index]
            if model_id not in self._exhausted:
                return model_id
        return self.model_ids[-1]

    def _get_model(self, model_id: str) -> GroqLiteLLMModel:
        if model_id not in self._models:
            quiet = len(self._models) > 0
            self._models[model_id] = GroqLiteLLMModel(
                model_id=model_id,
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=self.temperature,
                quiet=quiet,
            )
        return self._models[model_id]

    def _advance_after_failure(self, model_id: str, error: BaseException) -> bool:
        self._exhausted.add(model_id)
        while self._active_index < len(self.model_ids):
            if self.model_ids[self._active_index] not in self._exhausted:
                break
            self._active_index += 1

        reason = "unavailable" if is_groq_unavailable_model_error(error) else "limit hit"
        next_index = self._active_index
        while next_index < len(self.model_ids):
            candidate = self.model_ids[next_index]
            if candidate not in self._exhausted:
                print(
                    f"Groq {reason} on {model_id!r}. "
                    f"Switching to {candidate!r}."
                )
                self._active_index = next_index
                self.model_id = candidate
                return True
            next_index += 1

        print(f"Groq {reason} on {model_id!r} and no fallback models remain.")
        return False

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        last_error: BaseException | None = None
        attempts = 0
        max_attempts = len(self.model_ids)

        while attempts < max_attempts:
            model_id = self.active_model_id
            if model_id in self._exhausted:
                attempts += 1
                self._active_index += 1
                continue

            self.model_id = model_id
            try:
                return self._get_model(model_id).generate(
                    messages,
                    stop_sequences=stop_sequences,
                    response_format=response_format,
                    tools_to_call_from=tools_to_call_from,
                    **kwargs,
                )
            except Exception as error:
                last_error = error
                if not is_groq_fallback_error(error):
                    raise
                if not self._advance_after_failure(model_id, error):
                    raise
                attempts += 1

        if last_error:
            raise last_error
        raise RuntimeError("All Groq models in the fallback chain are exhausted")
