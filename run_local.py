#!/usr/bin/env python3
"""Local runner and Level 1 pass checker for the course GAIA subset."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent import GaiaAgent
from eval.course_client import (
    fetch_course_questions,
    fetch_random_question,
    submit_answers,
)
from file_resolver import resolve_task_attachment
from scoring import course_score_percent

API_URL = os.getenv(
    "COURSE_API_URL", "https://agents-course-unit4-scoring.hf.space"
)
PASS_TARGET = 6  # 30% of 20 Level 1 course questions


def run_question(
    agent: GaiaAgent,
    item: dict,
    download_dir: Path,
) -> str:
    file_path = None
    file_error = None
    file_name = item.get("file_name") or ""
    if file_name:
        file_path, file_error = resolve_task_attachment(
            API_URL, item["task_id"], file_name, download_dir
        )
    return agent(
        item["question"],
        file_path=file_path,
        file_error=file_error,
        task_id=item.get("task_id"),
    )


def print_score_summary(results: list[dict], submit_result: dict | None = None) -> None:
    if submit_result and submit_result.get("correct_count") is not None:
        correct = int(submit_result["correct_count"])
        total = int(submit_result.get("total_attempted") or len(results))
        score = float(submit_result.get("score", course_score_percent(correct, total)))
    else:
        correct = sum(1 for row in results if row["correct"])
        total = len(results)
        score = course_score_percent(correct, total)
    needed = PASS_TARGET

    print("\n" + "=" * 60)
    print(f"Level 1 course score: {score}% ({correct}/{total} correct)")
    print(f"Pass threshold: 30% ({needed}/{total} correct)")
    print("PASS" if score >= 30.0 else "NOT YET PASSING")
    print("=" * 60)

    print("\nResults:")
    for row in results:
        mark = "OK" if row.get("correct") else "X "
        expected = row.get("expected", "?")
        print(
            f"[{mark}] {row['task_id'][:8]}... "
            f"expected={expected!r} got={row['submitted']!r}"
        )


def grade_via_submit(results: list[dict], answers_payload: list[dict]) -> dict | None:
    username = os.getenv("HF_USERNAME", "").strip()
    if not username:
        print("HF_USERNAME not set — cannot grade via course API.")
        for row in results:
            row["correct"] = False
            row["expected"] = "?"
        return None

    agent_code = os.getenv(
        "AGENT_CODE",
        "https://huggingface.co/spaces/ken2ki/Final_Assignment_Template/tree/main",
    )
    submit_result = submit_answers(API_URL, username, agent_code, answers_payload)
    detail_map = {
        row.get("task_id"): row
        for row in submit_result.get("details") or submit_result.get("results") or []
    }
    if detail_map:
        for row in results:
            detail = detail_map.get(row["task_id"], {})
            row["correct"] = detail.get("correct", False)
            row["expected"] = detail.get("expected_answer", "?")
    print(
        f"Course API score: {submit_result.get('score', '?')}% "
        f"({submit_result.get('correct_count', '?')} correct)"
    )
    if submit_result.get("message"):
        print(f"Submit message: {submit_result['message']}")
    return submit_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the GAIA Level 1 agent locally against the course subset."
    )
    parser.add_argument(
        "--mode",
        choices=("single", "random", "all", "score"),
        default="score",
        help=(
            "single=one easy puzzle; random=one course question; "
            "all=run all 20 without grading; score=run all 20 and grade via course API"
        ),
    )
    args = parser.parse_args()
    agent = GaiaAgent()

    if args.mode == "single":
        answer = agent(
            'If you understand this sentence, write the opposite of the word "left" as the answer.',
            task_id="smoke-left",
        )
        print("\nANSWER:", answer)
        return

    if args.mode == "random":
        question = fetch_random_question(API_URL)
        with tempfile.TemporaryDirectory(prefix="gaia_files_") as temp_dir:
            answer = run_question(agent, question, Path(temp_dir))
        print("\nQUESTION:", question["question"])
        print("ANSWER:", answer)
        return

    questions = fetch_course_questions(API_URL)
    results: list[dict] = []
    answers_payload: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="gaia_files_") as temp_dir:
        download_dir = Path(temp_dir)
        for index, item in enumerate(questions, start=1):
            task_id = item["task_id"]
            print(f"\n[{index}/{len(questions)}] {item['question'][:90]}...")
            try:
                submitted = run_question(agent, item, download_dir)
                results.append(
                    {
                        "task_id": task_id,
                        "submitted": submitted,
                        "expected": "?",
                        "correct": False,
                    }
                )
                answers_payload.append(
                    {"task_id": task_id, "submitted_answer": submitted}
                )
                print(f"ANSWER: {submitted!r}")
            except Exception as error:
                results.append(
                    {
                        "task_id": task_id,
                        "submitted": f"ERROR: {error}",
                        "expected": "?",
                        "correct": False,
                    }
                )
                print("ERROR:", error)

    if args.mode == "score":
        submit_result = grade_via_submit(results, answers_payload)
        print_score_summary(results, submit_result)


if __name__ == "__main__":
    main()
