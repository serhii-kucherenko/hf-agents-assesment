"""GAIA benchmark agent built with smolagents."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from smolagents import CodeAgent

load_dotenv()

from model_provider import build_model, get_llm_provider, use_markdown_code_blocks
from scoring import normalize_answer
from tools import build_tools

MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))

# Adapted from the official GAIA leaderboard prompt:
# https://huggingface.co/spaces/gaia-benchmark/leaderboard
GAIA_SYSTEM_PROMPT = """
You are a general AI assistant solving GAIA Level 1 questions.

Use tools and Python code whenever needed:
- web search or Wikipedia for factual questions
- file tools for attachments
- get_youtube_transcript for YouTube links

Every action must be valid Python inside a code block.
When done, call final_answer("...") with ONLY the answer value.

Answer formatting rules (critical):
- Return a number, as few words as possible, or a comma-separated list
- Numbers: no commas in the number; no $ or % unless the question asks for them
- Strings: no articles; no abbreviations unless specified
- Lists: apply the same rules to each item
- Do NOT pass "FINAL ANSWER:" into final_answer()
- Do NOT add explanations, markdown, or extra punctuation

Examples:
- Question asks for a count -> final_answer("3")
- Question asks for a name -> final_answer("Smith")
- Question asks for a list -> final_answer("a, b, c") with a comma and space between items
""".strip()


def _extract_answer(raw_result: str) -> str:
    text = str(raw_result).strip()

    final_match = re.search(
        r'final_answer\s*\(\s*["\'](.+?)["\']\s*\)',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if final_match:
        return normalize_answer(final_match.group(1))

    if len(text) < 200 and "def " not in text and "import " not in text:
        return normalize_answer(text)

    if "```" in text:
        code_blocks = re.findall(r"```(?:\w*\n)?(.*?)```", text, flags=re.DOTALL)
        for block in reversed(code_blocks):
            block_match = re.search(
                r'final_answer\s*\(\s*["\'](.+?)["\']\s*\)',
                block,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if block_match:
                return normalize_answer(block_match.group(1))

    return normalize_answer(text)


def _build_prompt(question: str, file_path: str | None, file_error: str | None = None) -> str:
    parts = [question]

    if file_path:
        path = Path(file_path)
        suffix = path.suffix.lower()
        attachment_hint = {
            ".py": "A Python file is attached. Use read_text_file, then run or reason over it.",
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
            "Note: the benchmark file attachment could not be loaded "
            f"({file_error}). Answer using web search and other tools instead."
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
