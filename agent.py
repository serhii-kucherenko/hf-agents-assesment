"""GAIA benchmark agent built with smolagents."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from smolagents import CodeAgent

load_dotenv()

from model_provider import build_model, get_llm_provider, use_markdown_code_blocks
from tools import build_tools

MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))

GAIA_SYSTEM_PROMPT = """
You are an expert research assistant solving GAIA benchmark questions.

Rules:
- Use tools and Python code whenever needed.
- Search the web or Wikipedia for factual questions.
- For attached files, use the matching tool before answering.
- For YouTube links, use get_youtube_transcript first.
- Every action must be valid Python inside a code block.
- End with final_answer("your exact answer") once you are confident.
- Never output bare answers or partial tags outside a code block.
- Your final answer must contain ONLY the exact answer requested.
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


def _build_prompt(question: str, file_path: str | None, file_error: str | None = None) -> str:
    parts = [question]

    if file_path:
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
        parts.extend([attachment_hint, f"Attached file path: {path.resolve()}"])
    elif file_error:
        parts.append(
            "Note: the benchmark file attachment could not be downloaded from the "
            f"course API ({file_error}). Answer using web search and other tools instead."
        )

    return "\n\n".join(parts)


class GaiaAgent:
    def __init__(self):
        model = build_model()
        code_block_tags = "markdown" if use_markdown_code_blocks() else None
        self.agent = CodeAgent(
            tools=build_tools(),
            model=model,
            instructions=GAIA_SYSTEM_PROMPT,
            max_steps=MAX_STEPS,
            verbosity_level=1,
            code_block_tags=code_block_tags,
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
        print(
            f"GaiaAgent ready ({get_llm_provider()}, "
            f"code_blocks={'markdown' if code_block_tags else 'xml'})"
        )

    def __call__(
        self,
        question: str,
        file_path: str | None = None,
        file_error: str | None = None,
    ) -> str:
        prompt = _build_prompt(question, file_path, file_error)
        print(f"Running agent on question (first 80 chars): {prompt[:80]}...")
        result = self.agent.run(prompt)
        answer = _extract_answer(result)
        print(f"Agent answer: {answer[:120]}")
        return answer
