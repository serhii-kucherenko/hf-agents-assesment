from unittest.mock import MagicMock, patch

import pytest
from smolagents.models import ChatMessage, MessageRole

from provider_chain import (
    GROQ_API_BASE,
    ProviderFallbackModel,
    build_provider_fallback_chain,
    provider_fallback_order,
)


def _normalize(model_name: str) -> str:
    return model_name.strip()


def test_build_provider_chain_groq_cerebras_google():
    env = {
        "GROQ_API_KEY": "gsk_test",
        "CEREBRAS_API_KEY": "csk_test",
        "GOOGLE_API_KEY": "g_test",
        "GROQ_MODEL": "scout",
        "GROQ_MODEL_FALLBACK_CHAIN": "compound",
        "PROVIDER_FALLBACK_ORDER": "groq,cerebras,google",
    }
    with patch.dict("os.environ", env, clear=False):
        chain = build_provider_fallback_chain(_normalize)

    labels = [slot.label for slot in chain]
    assert labels[0] == "groq:scout"
    assert "groq:compound" in labels
    assert "cerebras:qwen-3-32b" in labels
    assert labels[-1] == "google:gemini/gemini-2.5-flash-lite"


def test_groq_stays_first_when_order_puts_cerebras_first():
    env = {
        "GROQ_API_KEY": "gsk_test",
        "CEREBRAS_API_KEY": "csk_test",
        "GOOGLE_API_KEY": "g_test",
        "PROVIDER_FALLBACK_ORDER": "cerebras,google,groq",
    }
    with patch.dict("os.environ", env, clear=False):
        chain = build_provider_fallback_chain(_normalize)

    assert chain[0].provider == "groq"
    assert any(slot.provider == "cerebras" for slot in chain)


def test_cerebras_model_id_uses_api_base_not_litellm_prefix():
    from provider_chain import _normalize_cerebras_model_id

    assert _normalize_cerebras_model_id("qwen-3-32b") == "qwen-3-32b"


def test_reset_for_question_clears_exhaustion():
    from provider_chain import ModelSlot

    model = ProviderFallbackModel(
        slots=[
            ModelSlot(provider="groq", model_id="scout", api_key="k", api_base=GROQ_API_BASE),
            ModelSlot(provider="google", model_id="gemini/x", api_key="k", api_base=None),
        ],
    )
    model._exhausted.add("groq:scout")
    model._active_index = 1
    model.reset_for_question()
    assert model._exhausted == set()
    assert model._active_index == 0
    assert model.model_id == "scout"


def test_provider_fallback_model_switches_on_litellm_provider_error():
    primary = MagicMock()
    fallback = MagicMock()
    primary.generate.side_effect = RuntimeError(
        "LLM Provider NOT provided. You passed model=qwen-3-32b"
    )
    fallback.generate.return_value = ChatMessage(role=MessageRole.ASSISTANT, content="ok")

    from provider_chain import ModelSlot

    model = ProviderFallbackModel(
        slots=[
            ModelSlot(
                provider="cerebras",
                model_id="qwen-3-32b",
                api_key="csk",
                api_base="https://api.cerebras.ai/v1",
            ),
            ModelSlot(
                provider="groq",
                model_id="scout",
                api_key="gsk",
                api_base="https://api.groq.com/openai/v1",
            ),
        ],
    )

    with patch.object(model, "_get_model", side_effect=lambda slot: {
        "cerebras:qwen-3-32b": primary,
        "groq:scout": fallback,
    }[slot.label]):
        result = model.generate([{"role": "user", "content": "hello"}])

    assert result.content == "ok"
    assert model.active_model_id == "scout"


def test_provider_fallback_order_normalizes_gemini():
    with patch.dict("os.environ", {"PROVIDER_FALLBACK_ORDER": "cerebras,gemini,groq"}, clear=False):
        assert provider_fallback_order() == ["cerebras", "google", "groq"]


def test_provider_fallback_model_switches_provider():
    primary = MagicMock()
    fallback = MagicMock()
    primary.generate.side_effect = RuntimeError("tokens per day (TPD): Limit 100000")
    fallback.generate.return_value = ChatMessage(role=MessageRole.ASSISTANT, content="ok")

    from provider_chain import ModelSlot

    model = ProviderFallbackModel(
        slots=[
            ModelSlot(
                provider="groq",
                model_id="scout",
                api_key="gsk",
                api_base="https://api.groq.com/openai/v1",
            ),
            ModelSlot(
                provider="cerebras",
                model_id="qwen-3-32b",
                api_key="csk",
                api_base="https://api.cerebras.ai/v1",
            ),
        ],
    )

    with patch.object(model, "_get_model", side_effect=lambda slot: {
        "groq:scout": primary,
        "cerebras:qwen-3-32b": fallback,
    }[slot.label]):
        result = model.generate([{"role": "user", "content": "hello"}])

    assert result.content == "ok"
    assert model.active_model_id == "qwen-3-32b"
