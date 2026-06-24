#!/usr/bin/env python3
"""Quick local test runner for the GAIA agent."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv

from agent import GaiaAgent

load_dotenv()

API_URL = "https://agents-course-unit4-scoring.hf.space"


def download_file(task_id: str, file_name: str, directory: Path) -> str:
    destination = directory / file_name
    response = requests.get(f"{API_URL}/files/{task_id}", timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return str(destination.resolve())


def run_question(agent: GaiaAgent, item: dict, download_dir: Path | None) -> str:
    file_path = None
    file_name = item.get("file_name") or ""
    if file_name and download_dir is not None:
        file_path = download_file(item["task_id"], file_name, download_dir)
    return agent(item["question"], file_path=file_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GAIA agent locally.")
    parser.add_argument(
        "--mode",
        choices=("single", "random", "all"),
        default="random",
        help="single=one easy puzzle, random=one API question, all=full 20-question run",
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
        print("\nQUESTION:", question["question"])
        print("ANSWER:", answer)
        return

    questions = requests.get(f"{API_URL}/questions", timeout=15).json()
    with tempfile.TemporaryDirectory(prefix="gaia_files_") as temp_dir:
        download_dir = Path(temp_dir)
        for index, item in enumerate(questions, start=1):
            print(f"\n[{index}/{len(questions)}] {item['question'][:90]}...")
            try:
                answer = run_question(agent, item, download_dir)
                print("ANSWER:", answer)
            except Exception as error:
                print("ERROR:", error)


if __name__ == "__main__":
    main()
