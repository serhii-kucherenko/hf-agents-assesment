#!/usr/bin/env python3
"""Local runner and Level 1 pass checker for the course GAIA subset."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv

from agent import GaiaAgent
from file_resolver import resolve_task_attachment
from gaia_data import get_ground_truth
from scoring import (
    course_answer_matches,
    course_score_percent,
    passes_course_threshold,
)

load_dotenv()

API_URL = "https://agents-course-unit4-scoring.hf.space"
PASS_TARGET = 6  # 30% of 20 Level 1 course questions


def fetch_course_questions() -> list[dict]:
    response = requests.get(f"{API_URL}/questions", timeout=15)
    response.raise_for_status()
    return response.json()


def run_question(agent: GaiaAgent, item: dict, download_dir: Path) -> str:
    file_path = None
    file_error = None
    file_name = item.get("file_name") or ""
    if file_name:
        file_path, file_error = resolve_task_attachment(
            API_URL, item["task_id"], file_name, download_dir
        )
    return agent(item["question"], file_path=file_path, file_error=file_error)


def print_score_summary(results: list[dict]) -> None:
    correct = sum(1 for row in results if row["correct"])
    total = len(results)
    score = course_score_percent(correct, total)
    needed = PASS_TARGET

    print("\n" + "=" * 60)
    print(f"Level 1 course score: {score}% ({correct}/{total} correct)")
    print(f"Pass threshold: 30% ({needed}/{total} correct)")
    print("PASS" if passes_course_threshold(correct, total) else "NOT YET PASSING")
    print("=" * 60)

    print("\nResults:")
    for row in results:
        mark = "OK" if row["correct"] else "X "
        print(
            f"[{mark}] {row['task_id'][:8]}... "
            f"expected={row['expected']!r} got={row['submitted']!r}"
        )


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
            "all=run all 20 without grading; score=run all 20 and grade locally"
        ),
    )
    args = parser.parse_args()
    agent = GaiaAgent()

    if args.mode == "single":
        answer = agent(
            'If you understand this sentence, write the opposite of the word "left" as the answer.'
        )
        print("\nANSWER:", answer)
        return

    if args.mode == "random":
        question = requests.get(f"{API_URL}/random-question", timeout=15).json()
        with tempfile.TemporaryDirectory(prefix="gaia_files_") as temp_dir:
            answer = run_question(agent, question, Path(temp_dir))
        expected = get_ground_truth(question["task_id"])
        print("\nQUESTION:", question["question"])
        print("ANSWER:", answer)
        if expected is not None:
            print("EXPECTED:", expected)
            print("MATCH:", course_answer_matches(answer, expected))
        return

    questions = fetch_course_questions()
    results: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="gaia_files_") as temp_dir:
        download_dir = Path(temp_dir)
        for index, item in enumerate(questions, start=1):
            task_id = item["task_id"]
            expected = get_ground_truth(task_id) or "?"
            print(f"\n[{index}/{len(questions)}] {item['question'][:90]}...")
            try:
                submitted = run_question(agent, item, download_dir)
                correct = (
                    expected != "?"
                    and course_answer_matches(submitted, expected)
                )
                results.append(
                    {
                        "task_id": task_id,
                        "submitted": submitted,
                        "expected": expected,
                        "correct": correct,
                    }
                )
                print(f"ANSWER: {submitted!r} | expected: {expected!r} | {'OK' if correct else 'MISS'}")
            except Exception as error:
                results.append(
                    {
                        "task_id": task_id,
                        "submitted": f"ERROR: {error}",
                        "expected": expected,
                        "correct": False,
                    }
                )
                print("ERROR:", error)

    if args.mode == "score":
        print_score_summary(results)


if __name__ == "__main__":
    main()
