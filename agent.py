"""GAIA benchmark agent built with smolagents."""

from __future__ import annotations

import os
import re
from pathlib import Path

from smolagents import CodeAgent, InferenceClientModel

from tools import build_tools

DEFAULT_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "12"))

GAIA_SYSTEM_PROMPT = """
You are an expert research assistant solving GAIA benchmark questions.

Rules:
- Use tools and Python code whenever needed.
- Search the web or Wikipedia for factual questions.
- For attached files, use the matching tool before answering.
- For YouTube links, use get_youtube_transcript.
- Think carefully, verify facts, and compute when needed.
- Your final response must contain ONLY the exact answer requested.
- Do not include explanations, markdown, prefixes, or the phrase "FINAL ANSWER".
- Match the requested format exactly: numbers, names, comma-separated lists, algebraic notation, etc.
- If asked for a comma-separated list, use commas with no extra spaces unless the question asks otherwise.
- If asked for a single word or name, return only that word or name.
""".strip()


def _extract_answer(raw_result: str) -> str:
    text = str(raw_result).strip()
    text = re.sub(r"^FINAL ANSWER:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Answer:\s*", "", text, flags=re.IGNORECASE)
    if "```" in text:
        code_blocks = re.findall(r"```(?:\w*\n)?(.*?)```", text, flags=re.DOTALL)
        if code_blocks:
            text = code_blocks[-1].strip()
    return text.strip()


def _build_prompt(question: str, file_path: str | None) -> str:
    if not file_path:
        return question

    path = Path(file_path)
    suffix = path.suffix.lower()
    attachment_hint = {
        ".py": "A Python file is attached. Use read_text_file if needed.",
        ".xlsx": "An Excel file is attached. Use read_excel_summary.",
        ".xls": "An Excel file is attached. Use read_excel_summary.",
        ".mp3": "An audio file is attached. Use transcribe_audio.",
        ".wav": "An audio file is attached. Use transcribe_audio.",
        ".png": "An image file is attached. Use describe_image.",
        ".jpg": "An image file is attached. Use describe_image.",
        ".jpeg": "An image file is attached. Use describe_image",
        ".webp": "An image file is attached. Use describe_image.",
    }.get(suffix, "A file is attached. Use the appropriate reading tool.")

    return (
        f"{question}\n\n"
        f"{attachment_hint}\n"
        f"Attached file path: {path.resolve()}"
    )


class GaiaAgent:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError(
                "Set HF_TOKEN in your Space secrets before running the evaluation."
            )

        model = InferenceClientModel(model_id=model_name, token=token)
        self.agent = CodeAgent(
            tools=build_tools(),
            model=model,
            max_steps=MAX_STEPS,
            verbosity_level=1,
            additional_authorized_imports=[
                "requests",
                "re",
                "json",
                "math",
                "statistics",
                "datetime",
                "collections",
                "itertools",
                "pandas",
                "numpy",
            ],
        )
        print(f"GaiaAgent initialized with model: {model_name}")

    def __call__(self, question: str, file_path: str | None = None) -> str:
        prompt = _build_prompt(question, file_path)
        print(f"Running agent on question (first 80 chars): {prompt[:80]}...")
        result = self.agent.run(prompt + "\n\n" + GAIA_SYSTEM_PROMPT)
        answer = _extract_answer(result)
        print(f"Agent answer: {answer[:120]}")
        return answer
