"""Model backend selection for local Ollama vs Hugging Face Inference."""

from __future__ import annotations

import os

from smolagents import InferenceClientModel, LiteLLMModel, Model


def get_llm_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in {"ollama", "hf"}:
        return explicit
    if os.getenv("USE_OLLAMA", "").strip().lower() in {"1", "true", "yes"}:
        return "ollama"
    if os.getenv("SPACE_ID"):
        return "hf"
    return "ollama"


def build_model() -> Model:
    provider = get_llm_provider()

    if provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "qwen3:14b")
        api_base = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
        ollama_model_id = (
            model_name
            if model_name.startswith("ollama_chat/")
            else f"ollama_chat/{model_name}"
        )
        print(
            f"Using Ollama model {ollama_model_id} at {api_base} "
            f"(num_ctx={num_ctx}, think=False)"
        )
        return LiteLLMModel(
            model_id=ollama_model_id,
            api_base=api_base,
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            num_ctx=num_ctx,
            think=False,
        )

    model_name = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Set HF_TOKEN for Hugging Face Inference, or run locally with "
            "USE_OLLAMA=1 in your .env file."
        )
    print(f"Using Hugging Face Inference model {model_name}")
    return InferenceClientModel(model_id=model_name, token=token)


def use_markdown_code_blocks() -> bool:
    if get_llm_provider() == "ollama":
        return True
    return os.getenv("AGENT_CODE_BLOCKS", "markdown").strip().lower() == "markdown"
