#!/usr/bin/env python3
"""Integration eval: fixtures, course API, or single task."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent import GaiaAgent
from agent.evolve import record_run, update_strategy_deltas
from eval.course_client import fetch_course_questions, submit_answers
from eval.report import build_report, print_report_summary
from file_resolver import resolve_task_attachment
from scoring import course_answer_matches, passes_course_threshold

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_API_URL = os.getenv(
    "COURSE_API_URL", "https://agents-course-unit4-scoring.hf.space"
)


def load_fixtures() -> list[dict]:
    items: list[dict] = []
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            items.extend(data)
    return items


def run_fixture(agent: GaiaAgent, fixture: dict) -> dict:
    answer = agent(
        fixture["question"],
        file_path=fixture.get("file_path"),
        task_id=fixture.get("task_id"),
    )
    expected = fixture["expected"]
    correct = course_answer_matches(answer, expected)
    return {
        "task_id": fixture.get("task_id", "fixture"),
        "question": fixture["question"],
        "submitted": answer,
        "expected": expected,
        "correct": correct,
        "question_tags": fixture.get("question_tags", []),
        "strategy": fixture.get("strategy", "direct"),
    }


def run_course(agent: GaiaAgent, api_url: str) -> list[dict]:
    questions = fetch_course_questions(api_url)
    results: list[dict] = []
    answers_payload: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="gaia_eval_") as temp_dir:
        download_dir = Path(temp_dir)
        for item in questions:
            task_id = item["task_id"]
            file_path = None
            file_error = None
            file_name = item.get("file_name") or ""
            if file_name:
                file_path, file_error = resolve_task_attachment(
                    api_url, task_id, file_name, download_dir
                )
            try:
                submitted = agent(
                    item["question"],
                    file_path=file_path,
                    file_error=file_error,
                    task_id=task_id,
                )
            except Exception as error:
                submitted = f"ERROR: {error}"
            answers_payload.append(
                {"task_id": task_id, "submitted_answer": submitted}
            )
            results.append(
                {
                    "task_id": task_id,
                    "question": item["question"],
                    "submitted": submitted,
                    "correct": False,
                }
            )

    username = os.getenv("HF_USERNAME", "").strip()
    if username:
        submit_result = submit_answers(
            api_url,
            username,
            os.getenv("AGENT_CODE", "local-eval"),
            answers_payload,
        )
        detail_map = {
            row.get("task_id"): row
            for row in submit_result.get("details") or submit_result.get("results") or []
        }
        correct_count = submit_result.get("correct_count", 0)
        for row in results:
            detail = detail_map.get(row["task_id"], {})
            row["correct"] = detail.get("correct", False)
            row["expected"] = detail.get("expected_answer", "?")
        print(f"Course submit score: {submit_result.get('score', '?')}% ({correct_count} correct)")
    else:
        print("HF_USERNAME not set — answers collected but not graded via course API.")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GAIA eval harness.")
    parser.add_argument(
        "--mode",
        choices=("fixtures", "course", "single"),
        default="fixtures",
    )
    parser.add_argument("--task-id", help="Fixture or course task id for single mode.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=float(os.getenv("EVAL_FIXTURE_MIN_SCORE", "80")),
        help="Minimum percent for fixtures mode pass gate.",
    )
    args = parser.parse_args()

    agent = GaiaAgent()
    results: list[dict] = []

    if args.mode == "fixtures":
        fixtures = load_fixtures()
        if args.task_id:
            fixtures = [f for f in fixtures if f.get("task_id") == args.task_id]
        for fixture in fixtures:
            print(f"Running fixture: {fixture.get('task_id')}")
            row = run_fixture(agent, fixture)
            results.append(row)
            record_run(
                row["task_id"],
                row["question"],
                row.get("strategy", "direct"),
                [],
                row["correct"],
            )
    elif args.mode == "course":
        results = run_course(agent, DEFAULT_API_URL)
    elif args.mode == "single":
        if not args.task_id:
            print("--task-id required for single mode", file=sys.stderr)
            sys.exit(1)
        fixtures = [f for f in load_fixtures() if f.get("task_id") == args.task_id]
        if fixtures:
            results = [run_fixture(agent, fixtures[0])]
        else:
            questions = fetch_course_questions(DEFAULT_API_URL)
            match = next((q for q in questions if q["task_id"] == args.task_id), None)
            if not match:
                print(f"Task {args.task_id} not found", file=sys.stderr)
                sys.exit(1)
            with tempfile.TemporaryDirectory(prefix="gaia_single_") as temp_dir:
                file_path = None
                file_error = None
                if match.get("file_name"):
                    file_path, file_error = resolve_task_attachment(
                        DEFAULT_API_URL,
                        match["task_id"],
                        match["file_name"],
                        Path(temp_dir),
                    )
                answer = agent(
                    match["question"],
                    file_path=file_path,
                    file_error=file_error,
                    task_id=match["task_id"],
                )
            results = [
                {
                    "task_id": match["task_id"],
                    "question": match["question"],
                    "submitted": answer,
                    "correct": False,
                }
            ]

    report_path = build_report(args.mode, results, RESULTS_DIR)
    update_strategy_deltas(report_path)
    correct = sum(1 for row in results if row.get("correct"))
    total = len(results)
    report = {
        "correct": correct,
        "total": total,
        "score_percent": round((correct / total) * 100, 2) if total else 0.0,
        "passed_threshold": passes_course_threshold(correct, total),
    }
    print_report_summary(report)

    if args.mode == "fixtures" and total:
        score = (correct / total) * 100
        if score < args.min_score:
            print(f"Fixture gate failed: {score:.1f}% < {args.min_score}%")
            sys.exit(1)


if __name__ == "__main__":
    main()
