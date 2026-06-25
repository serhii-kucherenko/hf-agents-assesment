"""Run the smolagents CodeAgent with all GAIA tools."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from smolagents import CodeAgent

load_dotenv()

from agent.tools import build_agent_tools
from agent.think_mode import should_use_think_mode
from model_provider import (
    apply_think_mode,
    build_model,
    get_llm_provider,
    supports_think_toggle,
    use_markdown_code_blocks,
)

MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "12"))

GAIA_SYSTEM_PROMPT = """
You are a general AI assistant solving GAIA Level 1 questions.

Use tools and Python code whenever needed:
- web search or Wikipedia for factual questions
- arxiv_search for academic papers
- file tools for attachments (PDF, Excel, CSV, images, audio)
- get_youtube_transcript for YouTube links
- execute_code for local Python or bash computation

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


def build_prompt(
    question: str,
    file_path: str | None,
    file_error: str | None = None,
    extra_sections: str | None = None,
) -> str:
    parts = [question]

    if file_path:
        path = Path(file_path)
        suffix = path.suffix.lower()
        attachment_hint = {
            ".py": "A Python file is attached. Use read_text_file, then run or reason over it.",
            ".xlsx": "An Excel file is attached. Use read_excel_summary.",
            ".xls": "An Excel file is attached. Use read_excel_summary.",
            ".csv": "A CSV file is attached. Use analyze_csv_file.",
            ".pdf": "A PDF file is attached. Use read_pdf.",
            ".mp3": "An audio file is attached. Use transcribe_audio.",
            ".wav": "An audio file is attached. Use transcribe_audio.",
            ".png": "An image file is attached. Use describe_image or extract_text_from_image.",
            ".jpg": "An image file is attached. Use describe_image or extract_text_from_image.",
            ".jpeg": "An image file is attached. Use describe_image or extract_text_from_image.",
            ".webp": "An image file is attached. Use describe_image or extract_text_from_image.",
        }.get(suffix, "A file is attached. Use the appropriate reading tool.")
        parts.extend([attachment_hint, f"Attached file path: {path.resolve()}"])
    elif file_error:
        parts.append(
            "Note: the benchmark file attachment could not be loaded "
            f"({file_error}). Answer using web search and other tools instead."
        )

    if extra_sections:
        parts.append(extra_sections)

    return "\n\n".join(parts)


class AgentRunner:
    def __init__(self):
        model = build_model()
        code_block_tags = "markdown" if use_markdown_code_blocks() else None
        self.agent = CodeAgent(
            tools=build_agent_tools(),
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
            f"AgentRunner ready ({get_llm_provider()}, "
            f"code_blocks={'markdown' if code_block_tags else 'xml'})"
        )

    def run(
        self,
        prompt: str,
        question: str | None = None,
        file_path: str | None = None,
        think: bool | None = None,
    ) -> str:
        if think is None and question is not None:
            think = should_use_think_mode(question, file_path)
        if think is not None and supports_think_toggle():
            apply_think_mode(self.agent.model, think)
        print(f"Running agent on question (first 80 chars): {prompt[:80]}...")
        return str(self.agent.run(prompt))
