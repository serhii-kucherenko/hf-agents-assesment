"""Self-correction loop for full pipeline depth."""

from __future__ import annotations

import os

from agent.agent_runner import AgentRunner
from agent.planner import pipeline_depth
from agent.verifier import extract_answer, verify_answer


def self_correction_enabled() -> bool:
    return pipeline_depth() == "full"


def max_correction_rounds() -> int:
    return int(os.getenv("SELF_CORRECTION_ROUNDS", "3"))


def run_with_self_correction(
    runner: AgentRunner,
    prompt: str,
    question: str,
    file_path: str | None = None,
) -> tuple[str, str]:
    """Return (raw_trace, answer) after optional correction rounds."""
    if not self_correction_enabled():
        raw = runner.run(prompt, question=question, file_path=file_path)
        return raw, extract_answer(raw)

    best_raw = ""
    best_answer = ""
    best_issue_count = 10_000

    current_prompt = prompt
    for round_index in range(max_correction_rounds()):
        raw = runner.run(current_prompt, question=question, file_path=file_path)
        result = verify_answer(question, raw)
        issue_count = len(result.issues)
        if issue_count < best_issue_count:
            best_raw = raw
            best_answer = result.answer
            best_issue_count = issue_count
        if result.approved:
            return raw, result.answer
        if round_index + 1 >= max_correction_rounds():
            break
        issues_text = "; ".join(result.issues)
        current_prompt = (
            f"{prompt}\n\nYour previous answer had issues: {issues_text}\n"
            f"Previous attempt answer: {result.answer}\n"
            "Fix the issues and call final_answer with the corrected value only."
        )

    return best_raw, best_answer or extract_answer(best_raw)
