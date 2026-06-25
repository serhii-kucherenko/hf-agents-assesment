"""Model backend: llama.cpp (local), Hugging Face Inference (Space), Ollama (fallback)."""

from __future__ import annotations

import os

import requests
from smolagents import InferenceClientModel, LiteLLMModel, Model, OpenAIServerModel

from groq_model import GroqLiteLLMModel
from provider_chain import (
    ProviderFallbackModel,
    build_provider_fallback_chain,
    provider_fallback_enabled,
)


def get_llm_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in {"llamacpp", "llama_cpp"}:
        return "llamacpp"
    if explicit == "hf":
        return "hf"
    if explicit == "ollama":
        return "ollama"
    if explicit == "groq":
        return "groq"
    if explicit in {"cerebras", "google", "gemini"}:
        return "cerebras" if explicit == "cerebras" else "google"

    if os.getenv("USE_OLLAMA", "").strip().lower() in {"1", "true", "yes"}:
        return "ollama"

    # On HF Space: prefer direct Groq if key is set (avoids HF inference credits)
    if os.getenv("SPACE_ID"):
        if os.getenv("GROQ_API_KEY"):
            return "groq"
        if os.getenv("CEREBRAS_API_KEY"):
            return "cerebras"
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            return "google"
        return "hf"

    return "llamacpp"


def _resolve_llamacpp_model_id(api_base: str, configured_id: str) -> str:
    """Match llama-server's advertised model id (often the GGUF path)."""
    try:
        response = requests.get(f"{api_base.rstrip('/')}/models", timeout=5)
        response.raise_for_status()
        models = response.json().get("data", [])
        if not models:
            return configured_id
        if len(models) == 1:
            return models[0]["id"]
        for entry in models:
            model_id = entry.get("id", "")
            if configured_id in model_id or model_id.endswith(f"{configured_id}.gguf"):
                return model_id
    except requests.RequestException:
        pass
    return configured_id


def _normalize_groq_model_id(model_name: str) -> str:
    """Map common Groq model names to IDs accepted by api.groq.com."""
    aliases = {
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "llama-3.3-70b": "llama-3.3-70b-versatile",
        "llama3.3-70b-versatile": "llama-3.3-70b-versatile",
        "llama-4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-4-scout-17b": "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen3-32b": "qwen/qwen3-32b",
        "qwen3.6-27b": "qwen/qwen3.6-27b",
        "allam-2-7b": "allam-2-7b",
        "llama-3.1-8b": "llama-3.1-8b-instant",
    }
    cleaned = model_name.strip()
    return aliases.get(cleaned, cleaned)


def build_model() -> Model:
    provider = get_llm_provider()

    if provider == "llamacpp":
        api_base = os.getenv("LLAMA_CPP_API_BASE", "http://127.0.0.1:8080/v1")
        configured_id = os.getenv("LLAMA_CPP_MODEL_ID", "Qwen3-14B-Q4_K_M")
        model_id = _resolve_llamacpp_model_id(api_base, configured_id)
        api_key = os.getenv("LLAMA_CPP_API_KEY", "llama")
        print(f"Using llama.cpp model {model_id} at {api_base}")
        return OpenAIServerModel(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key,
            temperature=0,
        )

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
            f"(num_ctx={num_ctx}, think=per-question)"
        )
        return LiteLLMModel(
            model_id=ollama_model_id,
            api_base=api_base,
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            num_ctx=num_ctx,
            think=False,
        )

    if provider == "groq":
        slots = build_provider_fallback_chain(_normalize_groq_model_id)
        if not slots:
            raise RuntimeError(
                "Set GROQ_API_KEY in Space secrets (or .env locally). "
                "Get a free key at https://console.groq.com — "
                "this bypasses Hugging Face inference credits."
            )
        if provider_fallback_enabled() and len(slots) > 1:
            print("Using cloud LLM with cross-provider fallback")
            return ProviderFallbackModel(slots=slots, temperature=0)
        slot = slots[0]
        print(f"Using {slot.provider} model {slot.model_id}")
        kwargs: dict = {
            "model_id": slot.model_id,
            "api_key": slot.api_key,
            "temperature": 0,
        }
        if slot.api_base:
            kwargs["api_base"] = slot.api_base
        return GroqLiteLLMModel(**kwargs)

    if provider == "cerebras":
        slots = build_provider_fallback_chain(_normalize_groq_model_id)
        cerebras_slots = [slot for slot in slots if slot.provider == "cerebras"]
        if not cerebras_slots:
            raise RuntimeError(
                "Set CEREBRAS_API_KEY in Space secrets or .env. "
                "Get a key at https://cloud.cerebras.ai"
            )
        if provider_fallback_enabled() and len(cerebras_slots) > 1:
            return ProviderFallbackModel(slots=cerebras_slots, temperature=0)
        slot = cerebras_slots[0]
        print(f"Using Cerebras model {slot.model_id}")
        return GroqLiteLLMModel(
            model_id=slot.model_id,
            api_base=slot.api_base,
            api_key=slot.api_key,
            temperature=0,
        )

    if provider == "google":
        slots = build_provider_fallback_chain(_normalize_groq_model_id)
        google_slots = [slot for slot in slots if slot.provider == "google"]
        if not google_slots:
            raise RuntimeError(
                "Set GOOGLE_API_KEY or GEMINI_API_KEY in Space secrets or .env. "
                "Get a key at https://aistudio.google.com/apikey"
            )
        if provider_fallback_enabled() and len(google_slots) > 1:
            return ProviderFallbackModel(slots=google_slots, temperature=0)
        slot = google_slots[0]
        print(f"Using Google Gemini model {slot.model_id}")
        return GroqLiteLLMModel(
            model_id=slot.model_id,
            api_key=slot.api_key,
            temperature=0,
        )

    model_name = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Hugging Face inference credits exhausted or HF_TOKEN missing. "
            "Fix options: (1) Add GROQ_API_KEY Space secret for free Groq API, "
            "(2) run locally with LLM_PROVIDER=ollama, or (3) add HF PRO/credits."
        )
    print(f"Using Hugging Face Inference model {model_name}")
    return InferenceClientModel(model_id=model_name, token=token)


def build_verifier_model() -> Model:
    critic = os.getenv("CRITIC_MODEL", "").strip()
    if critic and critic != os.getenv("LLAMA_CPP_MODEL_ID", ""):
        if os.getenv("SPACE_ID") or os.getenv("LLM_PROVIDER", "").lower() == "hf":
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            return InferenceClientModel(model_id=critic, token=token)
    return build_model()


def use_markdown_code_blocks() -> bool:
    provider = get_llm_provider()
    if provider in {"ollama", "llamacpp"}:
        return True
    return os.getenv("AGENT_CODE_BLOCKS", "markdown").strip().lower() == "markdown"


def supports_think_toggle() -> bool:
    provider = get_llm_provider()
    return provider in {"ollama", "llamacpp"}


def apply_think_mode(model: Model, think: bool) -> None:
    """Set per-request thinking for Ollama (think) or OpenAI-compat (reasoning_effort)."""
    if not supports_think_toggle():
        return

    provider = get_llm_provider()
    if provider == "ollama":
        model.kwargs["think"] = think
        print(f"Think mode: {'on' if think else 'off'} (Ollama)")
        return

    # llama.cpp / OpenAI-compatible (e.g. Qwen3 via llama-server)
    if think:
        model.kwargs.pop("reasoning_effort", None)
        model.kwargs.pop("extra_body", None)
    else:
        model.kwargs["reasoning_effort"] = "none"
    print(f"Think mode: {'on' if think else 'off'} (llama.cpp / OpenAI-compat)")
