"""Decide when to enable model thinking/reasoning for a GAIA question."""

from __future__ import annotations

import os
import re
from pathlib import Path


def think_mode_setting() -> str:
    """auto | on | off"""
    return os.getenv("THINK_MODE", "auto").strip().lower()


def _has_attachment(file_path: str | None) -> bool:
    return bool(file_path and Path(file_path).exists())


def _attachment_needs_think(file_path: str | None) -> bool:
    if not file_path:
        return False
    suffix = Path(file_path).suffix.lower()
    return suffix in {".pdf", ".xlsx", ".xls", ".csv", ".py", ".png", ".jpg", ".jpeg", ".webp"}


def should_use_think_mode(question: str, file_path: str | None = None) -> bool:
    """Return True when extended reasoning helps; False for quick factual answers."""
    mode = think_mode_setting()
    if mode in {"on", "true", "1", "yes"}:
        return True
    if mode in {"off", "false", "0", "no"}:
        return False

    text = question.strip()
    lower = text.lower()

    # Fast path: trivial smoke / one-step wording
    simple_patterns = (
        r"opposite of the word",
        r"reply with only",
        r"write only",
        r"answer with the word",
        r"how many letters are in the word",
    )
    if any(re.search(pattern, lower) for pattern in simple_patterns):
        return False

    if len(text) < 60 and not _has_attachment(file_path):
        return False

    # Multi-step / computation / comparison
    complex_signals = (
        r"\bhow many\b",
        r"\bcalculate\b",
        r"\bcompute\b",
        r"\bsum\b",
        r"\baverage\b",
        r"\bcompare\b",
        r"\bdifference\b",
        r"\bpercentage\b",
        r"\bratio\b",
        r"\bstep[s]?\b",
        r"\banaly[sz]e\b",
        r"\bdetermine\b",
        r"\bexplain why\b",
        r"\bwhich (one|country|city|year)\b",
        r"\baccording to\b",
        r"\bbased on\b",
        r"\barxiv\b",
        r"\byoutube\b",
        r"\bspreadsheet\b",
        r"\bexcel\b",
        r"\bpdf\b",
    )
    if any(re.search(pattern, lower) for pattern in complex_signals):
        return True

    if _attachment_needs_think(file_path):
        return True

    if len(text) > 180:
        return True

    if lower.count("?") >= 2:
        return True

    return False
