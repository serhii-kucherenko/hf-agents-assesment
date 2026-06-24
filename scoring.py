"""Scoring helpers aligned with the course Unit 4 API."""

from __future__ import annotations

import re


def normalize_answer(answer: str) -> str:
    text = str(answer).strip()
    text = re.sub(r"^FINAL ANSWER:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Answer:\s*", "", text, flags=re.IGNORECASE)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text.strip()


def course_answer_matches(submitted: str, ground_truth: str) -> bool:
    """Same comparison as agents-course Unit4 scoring API."""
    return normalize_answer(submitted).lower() == normalize_answer(ground_truth).lower()


def course_score_percent(correct_count: int, total_questions: int = 20) -> float:
    if total_questions == 0:
        return 0.0
    return round((correct_count / total_questions) * 100, 2)


def passes_course_threshold(correct_count: int, total_questions: int = 20) -> bool:
    return course_score_percent(correct_count, total_questions) >= 30.0
