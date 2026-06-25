"""Persistent memory for strategy hints and runtime errors."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

MEMORY_DIR = Path(os.getenv("AGENT_MEMORY_DIR", "memory"))
EXPERIENCE_FILE = MEMORY_DIR / "experience.jsonl"
RUNTIME_ERRORS_FILE = MEMORY_DIR / "runtime_errors.jsonl"


def _memory_dir() -> Path:
    return Path(os.getenv("AGENT_MEMORY_DIR", "memory"))


def _experience_file() -> Path:
    return _memory_dir() / "experience.jsonl"


def _runtime_errors_file() -> Path:
    return _memory_dir() / "runtime_errors.jsonl"


def _ensure_dir() -> None:
    _memory_dir().mkdir(parents=True, exist_ok=True)


def append_experience(record: dict) -> None:
    _ensure_dir()
    with _experience_file().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_runtime_error(record: dict) -> None:
    _ensure_dir()
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with _runtime_errors_file().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_experience(limit: int = 200) -> list[dict]:
    path = _experience_file()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return records[-limit:]


def load_runtime_errors(limit: int = 50) -> list[dict]:
    path = _runtime_errors_file()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return records[-limit:]


def question_tags(question: str) -> list[str]:
    tags: list[str] = []
    lower = question.lower()
    if any(ext in lower for ext in (".xlsx", "excel", "spreadsheet")):
        tags.append("excel")
    if any(ext in lower for ext in (".pdf", "pdf")):
        tags.append("pdf")
    if any(ext in lower for ext in (".png", ".jpg", "image", "picture")):
        tags.append("image")
    if any(word in lower for word in ("calculate", "sum", "average", "how many")):
        tags.append("numeric")
    if "youtube" in lower or "arxiv" in lower:
        tags.append("lookup")
    if not tags:
        tags.append("general")
    return tags
