"""Course scoring API client (no GAIA dataset download)."""

from __future__ import annotations

import os
from pathlib import Path

import requests

DEFAULT_API_URL = os.getenv(
    "COURSE_API_URL", "https://agents-course-unit4-scoring.hf.space"
)


def fetch_course_questions(api_url: str = DEFAULT_API_URL) -> list[dict]:
    response = requests.get(f"{api_url.rstrip('/')}/questions", timeout=15)
    response.raise_for_status()
    return response.json()


def fetch_random_question(api_url: str = DEFAULT_API_URL) -> dict:
    response = requests.get(f"{api_url.rstrip('/')}/random-question", timeout=15)
    response.raise_for_status()
    return response.json()


def download_attachment(
    api_url: str,
    task_id: str,
    file_name: str,
    dest_dir: Path,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = dest_dir / file_name
    response = requests.get(
        f"{api_url.rstrip('/')}/files/{task_id}",
        timeout=60,
    )
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def submit_answers(
    api_url: str,
    username: str,
    agent_code: str,
    answers: list[dict],
) -> dict:
    payload = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers,
    }
    response = requests.post(
        f"{api_url.rstrip('/')}/submit",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()
