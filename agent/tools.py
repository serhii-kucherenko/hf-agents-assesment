"""Extended tools for the GAIA agent."""

from __future__ import annotations

import cmath
from smolagents import tool

from agent.code_interpreter import run_python
from tools import (
    build_search_tool,
    visit_webpage,
    describe_image,
    fetch_url_as_markdown,
    get_youtube_transcript,
    read_excel_summary,
    read_text_file,
    transcribe_audio,
    wikipedia_search,
    wikipedia_studio_albums,
)


@tool
def arxiv_search(query: str) -> str:
    """Search arXiv for academic papers and return short excerpts.

    Args:
        query: Search terms for arXiv.
    """
    try:
        import arxiv

        search = arxiv.Search(query=query, max_results=3)
        parts: list[str] = []
        for paper in search.results():
            parts.append(
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(author.name for author in paper.authors)}\n"
                f"Summary: {paper.summary[:1200]}"
            )
        if not parts:
            return f"No arXiv papers found for: {query}"
        return "\n\n---\n\n".join(parts)[:12000]
    except Exception as error:
        return f"arXiv search failed: {error}. Try web search instead."


@tool
def execute_code(code: str, language: str = "python") -> str:
    """Execute Python or bash code locally and return stdout or errors.

    Args:
        code: Source code to run.
        language: python or bash.
    """
    if language.lower() == "python":
        return run_python(code)
    from agent.code_interpreter import CodeInterpreter

    result = CodeInterpreter().execute_code(code, language="bash")
    if result["status"] == "success":
        return (result.get("stdout") or "OK")[:12000]
    return (result.get("stderr") or "bash failed")[:12000]


@tool
def read_pdf(file_path: str) -> str:
    """Extract text from a PDF file.

    Args:
        file_path: Path to the PDF.
    """
    try:
        import fitz

        doc = fitz.open(file_path)
        parts = [page.get_text() for page in doc]
        return "\n".join(parts)[:12000]
    except Exception as error:
        return f"Could not read PDF: {error}"


@tool
def extract_text_from_image(file_path: str) -> str:
    """Extract text from an image using OCR.

    Args:
        file_path: Path to png, jpg, jpeg, or webp image.
    """
    try:
        import pytesseract
        from PIL import Image

        text = pytesseract.image_to_string(Image.open(file_path))
        return text.strip()[:12000] or "OCR returned no text."
    except Exception as error:
        return f"OCR failed: {error}. Try describe_image instead."


@tool
def analyze_csv_file(file_path: str) -> str:
    """Load a CSV and return column names, row count, and summary statistics.

    Args:
        file_path: Path to the CSV file.
    """
    try:
        import pandas as pd

        frame = pd.read_csv(file_path)
        return (
            f"Rows: {len(frame)}, Columns: {', '.join(frame.columns)}\n\n"
            f"{frame.describe(include='all').to_string()}"
        )[:12000]
    except Exception as error:
        return f"Could not analyze CSV: {error}"


@tool
def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a.

    Args:
        a: Number to subtract from.
        b: Number to subtract.
    """
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b.

    Args:
        a: Numerator.
        b: Denominator (must not be zero).
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def power(a: float, b: float) -> float:
    """Raise a to the power b.

    Args:
        a: Base.
        b: Exponent.
    """
    return a**b


@tool
def square_root(a: float) -> float | complex:
    """Return the square root of a.

    Args:
        a: Non-negative number for real square root.
    """
    if a >= 0:
        return a**0.5
    return cmath.sqrt(a)


def build_agent_tools() -> list:
    tools = [
        build_search_tool(),
        visit_webpage,
        wikipedia_search,
        wikipedia_studio_albums,
        arxiv_search,
        read_text_file,
        read_excel_summary,
        analyze_csv_file,
        read_pdf,
        transcribe_audio,
        describe_image,
        extract_text_from_image,
        get_youtube_transcript,
        execute_code,
        add,
        subtract,
        multiply,
        divide,
        power,
        square_root,
    ]
    if _browser_enabled():
        browser_tool = _build_browser_tool()
        if browser_tool is not None:
            tools.append(browser_tool)
    return tools


def _browser_enabled() -> bool:
    import os

    return os.getenv("BROWSER_ENABLED", "0").strip().lower() in {"1", "true", "yes"}


def _build_browser_tool():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    @tool
    def browse_url(url: str) -> str:
        """Open a URL in headless Chromium and return visible page text.

        Args:
            url: Web page URL to open.
        """
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=30000)
                text = page.inner_text("body")
                browser.close()
                return text[:12000]
        except Exception as error:
            return f"Browser navigation failed: {error}"

    return browse_url
