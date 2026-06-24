"""Tools for the GAIA evaluation agent."""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import DuckDuckGoSearchTool, VisitWebpageTool, tool
from youtube_transcript_api import YouTubeTranscriptApi


def build_search_tool() -> DuckDuckGoSearchTool:
    return DuckDuckGoSearchTool()


def build_visit_webpage_tool() -> VisitWebpageTool:
    return VisitWebpageTool()


@tool
def wikipedia_search(query: str) -> str:
    """Search English Wikipedia and return the opening text of the best matching article.

    Args:
        query: Search terms, ideally a person, place, or topic name.
    """
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3,
        }
        search_response = requests.get(search_url, params=search_params, timeout=20)
        search_response.raise_for_status()
        results = search_response.json().get("query", {}).get("search", [])
        if not results:
            return f"No Wikipedia articles found for: {query}"

        snippets: list[str] = []
        for result in results[:3]:
            title = result["title"]
            extract_params = {
                "action": "query",
                "prop": "extracts",
                "explaintext": True,
                "exintro": False,
                "titles": title,
                "format": "json",
            }
            extract_response = requests.get(search_url, params=extract_params, timeout=20)
            extract_response.raise_for_status()
            pages = extract_response.json().get("query", {}).get("pages", {})
            page = next(iter(pages.values()), {})
            extract = page.get("extract", "")
            snippets.append(f"Title: {title}\n{extract[:4000]}")
        return "\n\n---\n\n".join(snippets)
    except Exception as error:
        return f"Wikipedia search failed: {error}"


@tool
def fetch_url_as_markdown(url: str) -> str:
    """Fetch a web page and return readable markdown text.

    Args:
        url: Full URL to fetch.
    """
    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        markdown_content = markdownify(response.text).strip()
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        return markdown_content[:12000]
    except RequestException as error:
        return f"Error fetching URL: {error}"


@tool
def read_text_file(file_path: str) -> str:
    """Read a local text, Python, CSV, or JSON file and return its contents.

    Args:
        file_path: Absolute or relative path to the file.
    """
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:12000]
    except Exception as error:
        return f"Could not read file: {error}"


@tool
def read_excel_summary(file_path: str) -> str:
    """Read an Excel workbook and return all sheets as markdown tables.

    Args:
        file_path: Path to an .xlsx or .xls file.
    """
    try:
        import pandas as pd

        workbook = pd.read_excel(file_path, sheet_name=None)
        parts: list[str] = []
        for sheet_name, frame in workbook.items():
            parts.append(f"Sheet: {sheet_name}\n{frame.to_markdown(index=False)}")
        return "\n\n".join(parts)[:12000]
    except Exception as error:
        return f"Could not read Excel file: {error}"


@tool
def transcribe_audio(file_path: str) -> str:
    """Transcribe a local audio file such as mp3 or wav.

    Args:
        file_path: Path to the audio file.
    """
    path = Path(file_path)
    if not path.exists():
        return f"Audio file not found: {file_path}"

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return "HF_TOKEN is required for audio transcription."

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=token)
        with path.open("rb") as audio_file:
            transcript = client.automatic_speech_recognition(
                audio_file.read(),
                model="openai/whisper-large-v3",
            )
        if isinstance(transcript, dict):
            return transcript.get("text", str(transcript))
        return str(transcript)
    except Exception as error:
        return f"Audio transcription failed: {error}"


@tool
def describe_image(file_path: str, question: str = "Describe this image in detail.") -> str:
    """Analyze a local image file and answer a question about it.

    Args:
        file_path: Path to a png, jpg, jpeg, or webp image.
        question: What you want to know about the image.
    """
    path = Path(file_path)
    if not path.exists():
        return f"Image file not found: {file_path}"

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return "HF_TOKEN is required for image analysis."

    try:
        import base64

        from huggingface_hub import InferenceClient

        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/png")

        image_bytes = path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"

        client = InferenceClient(token=token)
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as error:
        return f"Image analysis failed: {error}"


@tool
def get_youtube_transcript(video_url: str) -> str:
    """Fetch the transcript/captions for a YouTube video URL.

    Args:
        video_url: A YouTube watch URL or youtu.be link.
    """
    match = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", video_url)
    if not match:
        return "Could not extract a YouTube video id from the URL."

    video_id = match.group(1)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US", "en-GB"])
        text = " ".join(entry["text"] for entry in transcript)
        return text[:12000]
    except Exception as error:
        return f"YouTube transcript unavailable: {error}"


def build_tools() -> list:
    return [
        build_search_tool(),
        build_visit_webpage_tool(),
        wikipedia_search,
        fetch_url_as_markdown,
        read_text_file,
        read_excel_summary,
        transcribe_audio,
        describe_image,
        get_youtube_transcript,
    ]
