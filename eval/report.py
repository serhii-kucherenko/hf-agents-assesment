"""Eval report helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scoring import course_answer_matches, course_score_percent, passes_course_threshold


def build_report(
    mode: str,
    results: list[dict],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    correct = sum(1 for row in results if row.get("correct"))
    total = len(results)
    report = {
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "correct": correct,
        "total": total,
        "score_percent": course_score_percent(correct, total),
        "passed_threshold": passes_course_threshold(correct, total),
        "results": results,
    }
    path = output_dir / f"report_{mode}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def print_report_summary(report: dict) -> None:
    print("\n" + "=" * 60)
    print(
        f"Score: {report['score_percent']}% "
        f"({report['correct']}/{report['total']} correct)"
    )
    print("PASS" if report["passed_threshold"] else "NOT YET PASSING")
    print("=" * 60)
