"""Cross-provider LLM fallback: Cerebras → Google Gemini → Groq."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from smolagents.models import ChatMessage, Model

from groq_model import (
    GroqLiteLLMModel,
    _groq_failure_reason,
    build_groq_model_chain,
    groq_fallback_enabled,
    groq_min_request_interval,
    is_groq_fallback_error,
)

DEFAULT_CEREBRAS_MODEL = "qwen-3-32b"
DEFAULT_CEREBRAS_MODEL_CHAIN = (
    "qwen-3-32b",
    "llama-3.3-70b",
    "llama3.1-8b",
)
DEFAULT_GOOGLE_MODEL = "gemini-2.5-flash-lite"

GROQ_API_BASE = "https://api.groq.com/openai/v1"
CEREBRAS_API_BASE = "https://api.cerebras.ai/v1"

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (compatible; GAIAAgent/1.0; "
    "+https://huggingface.co/spaces/ken2ki/Final_Assignment_Template)"
)


@dataclass(frozen=True)
class ModelSlot:
    provider: str
    model_id: str
    api_key: str
    api_base: str | None = None

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model_id}"


def provider_fallback_enabled() -> bool:
    explicit = os.getenv("PROVIDER_FALLBACK_ENABLED", "").strip().lower()
    if explicit in {"0", "false", "no", "off"}:
        return False
    if explicit in {"1", "true", "yes", "on"}:
        return True
    return groq_fallback_enabled()


def provider_fallback_order() -> list[str]:
    raw = os.getenv("PROVIDER_FALLBACK_ORDER", "cerebras,google,groq").strip()
    order = [part.strip().lower() for part in raw.split(",") if part.strip()]
    normalized: list[str] = []
    for provider in order:
        if provider == "gemini":
            provider = "google"
        if provider not in normalized:
            normalized.append(provider)
    return normalized or ["cerebras", "google", "groq"]


def google_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or None


def _normalize_cerebras_model_id(model_name: str) -> str:
    """LiteLLM requires cerebras/ prefix (no custom api_base)."""
    aliases = {
        "llama-3.3-70b": "llama-3.3-70b",
        "llama3.3-70b": "llama-3.3-70b",
        "llama-3.1-8b": "llama3.1-8b",
        "llama3.1-8b": "llama3.1-8b",
        "qwen3-32b": "qwen-3-32b",
        "qwen-3-32b": "qwen-3-32b",
    }
    cleaned = model_name.strip()
    if cleaned.startswith("cerebras/"):
        cleaned = cleaned[len("cerebras/") :]
    cleaned = aliases.get(cleaned, cleaned)
    return f"cerebras/{cleaned}"


def _normalize_google_model_id(model_name: str) -> str:
    cleaned = model_name.strip()
    if cleaned.startswith("gemini/"):
        return cleaned
    return f"gemini/{cleaned}"


def _single_provider_slot(
    *,
    provider: str,
    model_id: str,
    api_key: str,
    api_base: str | None,
    normalize: Callable[[str], str],
) -> list[ModelSlot]:
    return [
        ModelSlot(
            provider=provider,
            model_id=normalize(model_id),
            api_key=api_key,
            api_base=api_base,
        )
    ]

def _groq_slots(normalize_groq: Callable[[str], str]) -> list[ModelSlot]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return []
    api_base = os.getenv("GROQ_API_BASE", GROQ_API_BASE)
    return [
        ModelSlot(provider="groq", model_id=model_id, api_key=api_key, api_base=api_base)
        for model_id in build_groq_model_chain(normalize_groq)
    ]


def _cerebras_slots(_normalize_groq: Callable[[str], str]) -> list[ModelSlot]:
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        return []
    model_ids = [_normalize_cerebras_model_id(model_id) for model_id in DEFAULT_CEREBRAS_MODEL_CHAIN]
    return [
        ModelSlot(provider="cerebras", model_id=model_id, api_key=api_key, api_base=None)
        for model_id in model_ids
    ]


def _google_slots(_normalize_groq: Callable[[str], str]) -> list[ModelSlot]:
    api_key = google_api_key()
    if not api_key:
        return []
    return _single_provider_slot(
        provider="google",
        model_id=DEFAULT_GOOGLE_MODEL,
        api_key=api_key,
        api_base=None,
        normalize=_normalize_google_model_id,
    )


def _cloud_api_keys_present() -> dict[str, bool]:
    return {
        "cerebras": bool(os.getenv("CEREBRAS_API_KEY")),
        "google": bool(google_api_key()),
        "groq": bool(os.getenv("GROQ_API_KEY")),
    }


def build_provider_fallback_chain(normalize_groq: Callable[[str], str]) -> list[ModelSlot]:
    builders = {
        "groq": _groq_slots,
        "cerebras": _cerebras_slots,
        "google": _google_slots,
    }
    slots: list[ModelSlot] = []
    seen: set[str] = set()
    for provider in provider_fallback_order():
        builder = builders.get(provider)
        if builder is None:
            continue
        for slot in builder(normalize_groq):
            if slot.label in seen:
                continue
            slots.append(slot)
            seen.add(slot.label)

    keys = _cloud_api_keys_present()
    missing = [name for name, present in keys.items() if not present]
    if missing:
        print(f"Cloud API keys not set (providers skipped): {', '.join(missing)}")
    return slots


class ProviderFallbackModel(Model):
    """Try cloud models across providers; switch on hard rate, quota, or context limits."""

    _last_request_at: float = 0.0

    def __init__(self, slots: list[ModelSlot], temperature: float = 0) -> None:
        if not slots:
            raise ValueError("Provider fallback chain cannot be empty")
        super().__init__(model_id=slots[0].model_id, temperature=temperature)
        self.slots = slots
        self.temperature = temperature
        self._models: dict[str, GroqLiteLLMModel] = {}
        self._active_index = 0
        self._exhausted: set[str] = set()
        print(f"Provider fallback chain: {' -> '.join(slot.label for slot in slots)}")

    def reset_for_question(self) -> None:
        """Each question gets a fresh rotation — exhaustion must not carry over."""
        self._exhausted.clear()
        self._active_index = 0
        if self.slots:
            self.model_id = self.slots[0].model_id

    @property
    def active_slot(self) -> ModelSlot:
        for index in range(self._active_index, len(self.slots)):
            slot = self.slots[index]
            if slot.label not in self._exhausted:
                return slot
        return self.slots[-1]

    @property
    def active_model_id(self) -> str:
        return self.active_slot.model_id

    def _get_model(self, slot: ModelSlot) -> GroqLiteLLMModel:
        if slot.label not in self._models:
            quiet = len(self._models) > 0
            kwargs: dict = {
                "model_id": slot.model_id,
                "api_key": slot.api_key,
                "temperature": self.temperature,
                "quiet": quiet,
            }
            if slot.api_base:
                kwargs["api_base"] = slot.api_base
            self._models[slot.label] = GroqLiteLLMModel(**kwargs)
        return self._models[slot.label]

    def _shared_throttle(self) -> None:
        interval = groq_min_request_interval()
        if interval <= 0:
            return
        now = time.time()
        elapsed = now - ProviderFallbackModel._last_request_at
        if elapsed < interval:
            time.sleep(interval - elapsed)
        ProviderFallbackModel._last_request_at = time.time()

    def _advance_after_failure(self, slot: ModelSlot, error: BaseException) -> bool:
        self._exhausted.add(slot.label)
        while self._active_index < len(self.slots):
            if self.slots[self._active_index].label not in self._exhausted:
                break
            self._active_index += 1

        reason = _groq_failure_reason(error)
        next_index = self._active_index
        while next_index < len(self.slots):
            candidate = self.slots[next_index]
            if candidate.label not in self._exhausted:
                print(
                    f"{slot.provider} {reason} on {slot.model_id!r}. "
                    f"Switching to {candidate.label!r}."
                )
                self._active_index = next_index
                self.model_id = candidate.model_id
                return True
            next_index += 1

        print(f"{slot.provider} {reason} on {slot.model_id!r} and no fallback slots remain.")
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
        max_attempts = len(self.slots)

        while attempts < max_attempts:
            slot = self.active_slot
            if slot.label in self._exhausted:
                attempts += 1
                self._active_index += 1
                continue

            self.model_id = slot.model_id
            try:
                self._shared_throttle()
                return self._get_model(slot).generate(
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
                if not self._advance_after_failure(slot, error):
                    raise
                attempts += 1

        if last_error:
            raise last_error
        raise RuntimeError("All models in the provider fallback chain are exhausted")


class GroqFallbackModel(ProviderFallbackModel):
    """Backward-compatible Groq-only fallback wrapper."""

    def __init__(
        self,
        model_ids: list[str],
        api_key: str,
        api_base: str,
        temperature: float = 0,
    ) -> None:
        self.model_ids = model_ids
        self.api_key = api_key
        self.api_base = api_base
        slots = [
            ModelSlot(provider="groq", model_id=model_id, api_key=api_key, api_base=api_base)
            for model_id in model_ids
        ]
        super().__init__(slots=slots, temperature=temperature)
