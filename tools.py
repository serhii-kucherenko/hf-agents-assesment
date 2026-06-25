"""Tools for the GAIA evaluation agent."""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import DuckDuckGoSearchTool, tool
from youtube_transcript_api import YouTubeTranscriptApi

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (compatible; GAIAAgent/1.0; "
    "+https://huggingface.co/spaces/ken2ki/Final_Assignment_Template)"
)
WIKIPEDIA_HEADERS = {
    "User-Agent": os.getenv("WIKIPEDIA_USER_AGENT", BROWSER_USER_AGENT),
}
FETCH_HEADERS = {"User-Agent": BROWSER_USER_AGENT}


def build_search_tool() -> DuckDuckGoSearchTool:
    return DuckDuckGoSearchTool()


@tool
def visit_webpage(url: str) -> str:
    """Fetch a web page and return readable markdown text.

    Args:
        url: Full URL to fetch.
    """
    return fetch_url_as_markdown(url)


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
        search_response = requests.get(
            search_url, params=search_params, timeout=20, headers=WIKIPEDIA_HEADERS
        )
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
            extract_response = requests.get(
                search_url, params=extract_params, timeout=20, headers=WIKIPEDIA_HEADERS
            )
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
        response = requests.get(url, timeout=30, headers=FETCH_HEADERS)
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

    try:
        from faster_whisper import WhisperModel

        model_size = os.getenv("WHISPER_MODEL", "base")
        whisper = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _info = whisper.transcribe(str(path))
        text = " ".join(segment.text.strip() for segment in segments)
        if text:
            return text[:12000]
    except Exception as local_error:
        hf_error = None
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if token:
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
                hf_error = error
        if hf_error:
            return (
                f"Local transcription failed: {local_error}. "
                f"HF fallback failed: {hf_error}"
            )
        return (
            f"Local transcription failed: {local_error}. "
            "Install faster-whisper or set HF_TOKEN for cloud fallback."
        )

    return "Audio transcription returned no text."


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

    import base64

    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    vision_model = os.getenv("OLLAMA_VISION_MODEL", "").strip()
    if vision_model:
        try:
            api_base = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")
            response = requests.post(
                f"{api_base.rstrip('/')}/api/chat",
                json={
                    "model": vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": question,
                            "images": [image_b64],
                        }
                    ],
                    "stream": False,
                },
                timeout=180,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as error:
            return f"Ollama vision analysis failed: {error}"

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return (
            "No local vision model configured. Set OLLAMA_VISION_MODEL in .env "
            "(for example after running `ollama pull llava:7b`) or set HF_TOKEN."
        )

    try:
        from huggingface_hub import InferenceClient

        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/png")
        data_url = f"data:{mime_type};base64,{image_b64}"

        client = InferenceClient(token=token)
        vision_model = os.getenv("HF_VISION_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")
        response = client.chat.completions.create(
            model=vision_model,
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

    If captions are unavailable, returns the video title plus web search results
    about the video so you can still infer the answer.

    Args:
        video_url: A YouTube watch URL or youtu.be link.
    """
    match = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", video_url)
    if not match:
        return "Could not extract a YouTube video id from the URL."

    video_id = match.group(1)
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        text = " ".join(snippet.text for snippet in transcript)
        return text[:12000]
    except Exception as transcript_error:
        try:
            oembed = requests.get(
                "https://www.youtube.com/oembed",
                params={"url": video_url, "format": "json"},
                timeout=20,
            )
            oembed.raise_for_status()
            title = oembed.json().get("title", video_id)
        except Exception:
            title = video_id

        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        f'"{title}" bird species video transcript summary',
                        max_results=5,
                    )
                )
            snippets = []
            for item in results:
                body = item.get("body") or item.get("title") or str(item)
                snippets.append(body)
            search_text = "\n\n".join(snippets)
        except Exception as search_error:
            search_text = f"Web search fallback failed: {search_error}"

        return (
            f"YouTube transcript unavailable ({transcript_error}).\n"
            f"Video title: {title}\n"
            f"Use the following web search results about the video instead:\n\n"
            f"{search_text[:10000]}"
        )


def build_tools() -> list:
    return [
        build_search_tool(),
        visit_webpage,
        wikipedia_search,
        fetch_url_as_markdown,
        read_text_file,
        read_excel_summary,
        transcribe_audio,
        describe_image,
        get_youtube_transcript,
    ]
