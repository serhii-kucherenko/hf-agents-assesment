from unittest.mock import MagicMock, patch

import pytest
from smolagents.models import ChatMessage, MessageRole

from groq_model import (
    GROQ_DEFAULT_SPACE_MODEL,
    GroqFallbackModel,
    build_groq_model_chain,
    call_with_groq_retry,
    default_groq_model_id,
    groq_min_request_interval,
    groq_requests_per_minute,
    groq_retry_wait_seconds,
    is_groq_quota_exhausted_error,
    is_groq_unavailable_model_error,
    is_groq_fallback_error,
    is_hard_groq_limit_error,
    parse_groq_retry_seconds,
)


def _normalize(model_name: str) -> str:
    return model_name.strip()


def test_default_groq_fallback_chain_includes_free_tier_models():
    with patch.dict("os.environ", {"GROQ_MODEL": "", "SPACE_ID": "user/space"}, clear=False):
        chain = build_groq_model_chain(_normalize)
    assert chain[0] == GROQ_DEFAULT_SPACE_MODEL
    assert "llama-3.1-8b-instant" in chain
    assert "openai/gpt-oss-20b" in chain
    assert "allam-2-7b" in chain
    assert len(chain) >= 8


def test_is_groq_unavailable_model_error():
    error = RuntimeError("The model `compound-mini` does not exist or you do not have access to it.")
    assert is_groq_unavailable_model_error(error)
    assert is_groq_fallback_error(error)


def test_groq_fallback_model_skips_missing_model():
    primary = MagicMock()
    fallback = MagicMock()
    primary.generate.side_effect = RuntimeError(
        "model_not_found: The model `compound-mini` does not exist"
    )
    fallback.generate.return_value = ChatMessage(role=MessageRole.ASSISTANT, content="ok")

    model = GroqFallbackModel(
        model_ids=["compound-mini", "llama-3.1-8b-instant"],
        api_key="test",
        api_base="https://api.groq.com/openai/v1",
    )

    with patch.object(model, "_get_model", side_effect=lambda model_id: {
        "compound-mini": primary,
        "llama-3.1-8b-instant": fallback,
    }[model_id]):
        result = model.generate([{"role": "user", "content": "hello"}])

    assert result.content == "ok"
    assert primary.generate.call_count == 1
    assert model.active_model_id == "llama-3.1-8b-instant"


def test_is_hard_groq_limit_error_rate_limit():
    assert is_hard_groq_limit_error(RuntimeError("429 rate limit reached"))


def test_is_hard_groq_limit_error_daily_quota():
    assert is_hard_groq_limit_error(RuntimeError("tokens per day limit exceeded"))


def test_is_hard_groq_limit_error_other():
    assert not is_hard_groq_limit_error(RuntimeError("connection reset"))


def test_build_groq_model_chain_dedupes_primary():
    with patch.dict(
        "os.environ",
        {
            "GROQ_MODEL": "meta-llama/llama-4-scout-17b-16e-instruct",
            "GROQ_MODEL_FALLBACK_CHAIN": "groq/compound-mini,llama-3.1-8b-instant",
        },
        clear=False,
    ):
        chain = build_groq_model_chain(_normalize)
    assert chain == [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "groq/compound-mini",
        "llama-3.1-8b-instant",
    ]


def test_groq_fallback_model_switches_after_hard_limit():
    primary = MagicMock()
    fallback = MagicMock()
    primary.generate.side_effect = RuntimeError("429 rate limit reached")
    fallback.generate.return_value = ChatMessage(role=MessageRole.ASSISTANT, content="42")

    model = GroqFallbackModel(
        model_ids=["scout", "compound"],
        api_key="test",
        api_base="https://api.groq.com/openai/v1",
    )

    with patch.object(model, "_get_model", side_effect=lambda model_id: {
        "scout": primary,
        "compound": fallback,
    }[model_id]):
        result = model.generate([{"role": "user", "content": "hello"}])

    assert result.content == "42"
    assert model.active_model_id == "compound"
    assert "scout" in model._exhausted


def test_groq_fallback_model_raises_when_chain_exhausted():
    primary = MagicMock()
    primary.generate.side_effect = RuntimeError("429 rate limit reached")

    model = GroqFallbackModel(
        model_ids=["only-model"],
        api_key="test",
        api_base="https://api.groq.com/openai/v1",
    )

    with patch.object(model, "_get_model", return_value=primary):
        with pytest.raises(RuntimeError, match="429"):
            model.generate([{"role": "user", "content": "hello"}])

def test_parse_groq_retry_seconds_hms():
    message = "Please try again in 1h7m12.288s."
    assert parse_groq_retry_seconds(message) == pytest.approx(1 * 3600 + 7 * 60 + 12.288 + 1)


def test_is_groq_quota_exhausted_error_tpd():
    error = RuntimeError(
        "Rate limit reached on tokens per day (TPD): Limit 100000, Used 97057"
    )
    assert is_groq_quota_exhausted_error(error)
    assert is_hard_groq_limit_error(error)


def test_call_with_groq_retry_skips_sleep_on_tpd():
    calls = {"count": 0}

    def tpd_error():
        calls["count"] += 1
        raise RuntimeError(
            "tokens per day (TPD): Limit 100000. Please try again in 1h7m12s."
        )

    with patch("groq_model.time.sleep") as sleep_mock:
        with pytest.raises(RuntimeError, match="tokens per day"):
            call_with_groq_retry(tpd_error)
    assert calls["count"] == 1
    sleep_mock.assert_not_called()


def test_groq_fallback_model_switches_on_tpd_without_retry():
    primary = MagicMock()
    fallback = MagicMock()
    primary.generate.side_effect = RuntimeError(
        "tokens per day (TPD): Limit 100000. Please try again in 1h7m12s."
    )
    fallback.generate.return_value = ChatMessage(role=MessageRole.ASSISTANT, content="ok")

    model = GroqFallbackModel(
        model_ids=["scout", "compound"],
        api_key="test",
        api_base="https://api.groq.com/openai/v1",
    )

    with patch.object(model, "_get_model", side_effect=lambda model_id: {
        "scout": primary,
        "compound": fallback,
    }[model_id]):
        result = model.generate([{"role": "user", "content": "hello"}])

    assert result.content == "ok"
    assert primary.generate.call_count == 1


def test_parse_groq_retry_seconds():
    message = (
        "Rate limit reached for model llama-3.3-70b-versatile ... "
        "Please try again in 13.535s."
    )
    assert parse_groq_retry_seconds(message) == pytest.approx(14.535)


def test_parse_groq_retry_seconds_missing():
    assert parse_groq_retry_seconds("some other error") is None


def test_groq_retry_wait_seconds_uses_api_hint():
    error = RuntimeError("Please try again in 10s")
    assert groq_retry_wait_seconds(error, attempt=1) == pytest.approx(11.0)


def test_groq_retry_wait_seconds_fallback():
    error = RuntimeError("429 too many requests")
    with patch.dict("os.environ", {"GROQ_RETRY_WAIT": "12"}, clear=False):
        assert groq_retry_wait_seconds(error, attempt=2) == pytest.approx(24.0)


def test_groq_min_request_interval_to_rpm():
    with patch.dict("os.environ", {"GROQ_MIN_REQUEST_INTERVAL": "3"}, clear=False):
        assert groq_min_request_interval() == 3.0
        assert groq_requests_per_minute() == pytest.approx(20.0)


def test_default_groq_model_id_space():
    with patch.dict(
        "os.environ",
        {"SPACE_ID": "user/space", "GROQ_MODEL": ""},
        clear=False,
    ):
        assert default_groq_model_id() == GROQ_DEFAULT_SPACE_MODEL


def test_default_groq_model_id_explicit_override():
    with patch.dict("os.environ", {"GROQ_MODEL": "llama-3.3-70b-versatile"}, clear=False):
        assert default_groq_model_id() == "llama-3.3-70b-versatile"


def test_call_with_groq_retry_recovers_after_rate_limit():
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("429 rate_limit_exceeded: try again in 0.01s")
        return "ok"

    with patch("groq_model.time.sleep"):
        assert call_with_groq_retry(flaky) == "ok"
    assert calls["count"] == 2


def test_call_with_groq_retry_raises_after_max_attempts():
    def always_rate_limited():
        raise RuntimeError("429 rate limit: try again in 0.01s")

    with patch.dict("os.environ", {"GROQ_MAX_RETRIES": "2"}, clear=False):
        with patch("groq_model.time.sleep"):
            with pytest.raises(RuntimeError, match="429"):
                call_with_groq_retry(always_rate_limited)
