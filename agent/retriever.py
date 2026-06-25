"""Strategy hints from past graded runs (never ground-truth answers)."""

from __future__ import annotations

import os

from agent.memory.store import load_experience, question_tags


def retriever_enabled() -> bool:
    return os.getenv("RETRIEVER_ENABLED", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def get_strategy_hints(question: str, top_k: int = 3) -> str:
    if not retriever_enabled():
        return ""

    tags = set(question_tags(question))
    scored: list[tuple[int, dict]] = []
    for record in load_experience():
        record_tags = set(record.get("question_tags") or [])
        overlap = len(tags & record_tags)
        if overlap == 0:
            continue
        scored.append((overlap, record))

    scored.sort(key=lambda item: (item[0], item[1].get("correct", False)), reverse=True)
    hints: list[str] = []
    for _, record in scored[:top_k]:
        strategy = record.get("strategy") or ""
        tools_used = record.get("tools_used") or []
        if strategy:
            hints.append(f"- Strategy: {strategy}")
        if tools_used:
            hints.append(f"- Tools that worked: {', '.join(tools_used)}")
    if not hints:
        return ""

    return "Past strategy hints (do not copy answers):\n" + "\n".join(hints)
