"""Post-eval memory updates (strategy hints only, never answers)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agent.memory.store import append_experience, question_tags


def self_evolve_enabled() -> bool:
    return os.getenv("SELF_EVOLVE", "0").strip().lower() in {"1", "true", "yes"}


def record_run(
    task_id: str,
    question: str,
    strategy: str,
    tools_used: list[str],
    correct: bool,
) -> None:
    if not self_evolve_enabled():
        return
    append_experience(
        {
            "task_id": task_id,
            "question_tags": question_tags(question),
            "strategy": strategy,
            "tools_used": tools_used,
            "correct": correct,
        }
    )


def update_strategy_deltas(report_path: Path) -> None:
    if not self_evolve_enabled() or not report_path.exists():
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    deltas: dict[str, dict] = {}
    for row in report.get("results") or []:
        if not row.get("correct"):
            continue
        for tag in row.get("question_tags") or []:
            entry = deltas.setdefault(tag, {"successes": 0, "strategies": []})
            entry["successes"] += 1
            strategy = row.get("strategy")
            if strategy and strategy not in entry["strategies"]:
                entry["strategies"].append(strategy)

    memory_dir = Path(os.getenv("AGENT_MEMORY_DIR", "memory"))
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "strategy_deltas.json").write_text(
        json.dumps(deltas, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
