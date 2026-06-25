"""Optional live Cerebras API tests — skipped unless CEREBRAS_API_KEY is set."""

from __future__ import annotations

import os

import pytest

from provider_chain import CloudLiteLLMModel, build_cerebras_model_chain, _normalize_cerebras_model_id

pytestmark = pytest.mark.skipif(
    not (os.getenv("CEREBRAS_API_KEY") or "").strip(),
    reason="CEREBRAS_API_KEY not set",
)


@pytest.mark.parametrize("raw_model", build_cerebras_model_chain())
def test_cerebras_model_responds(raw_model: str) -> None:
    model_id = _normalize_cerebras_model_id(raw_model)
    model = CloudLiteLLMModel(
        "cerebras",
        model_id=model_id,
        api_key=os.environ["CEREBRAS_API_KEY"],
        temperature=0,
        quiet=True,
    )
    reply = model.generate([{"role": "user", "content": "Reply with exactly: ok"}])
    assert reply.content
