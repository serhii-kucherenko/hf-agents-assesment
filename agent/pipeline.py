"""GAIA pipeline: retriever → plan → agent → verify → answer."""

from __future__ import annotations

import os
import re

from agent.agent_runner import AgentRunner, build_prompt
from agent.artifacts.writer import write_evidence, write_notes
from agent.planner import create_plan, pipeline_depth, plan_to_prompt_section, should_plan
from agent.retriever import get_strategy_hints
from agent.think_mode import should_use_think_mode
from agent.self_correction import run_with_self_correction
from agent.verifier import extract_answer, verify_answer
from agent.voting import majority_vote, vote_runs


def _double_compute_hint(question: str, answer: str) -> list[str]:
    """Standard depth: flag numeric answers for independent recomputation."""
    if pipeline_depth() not in {"standard", "full"}:
        return []
    if re.search(r"\b(how many|sum|total|average|calculate)\b", question, re.I):
        if re.fullmatch(r"-?\d+(?:\.\d+)?", answer.strip()):
            return ["Recompute the numeric result with execute_code before final_answer."]
    return []


def run_pipeline(
    runner: AgentRunner,
    question: str,
    file_path: str | None = None,
    file_error: str | None = None,
    task_id: str | None = None,
) -> tuple[str, str]:
    """Return (answer, raw_trace)."""
    strategy_hints = get_strategy_hints(question)
    file_hint = file_path or ""
    plan = create_plan(question, file_hint or None, strategy_hints, task_id=task_id)
    plan_section = plan_to_prompt_section(plan)

    extra_sections = []
    if strategy_hints:
        extra_sections.append(strategy_hints)
    if plan_section:
        extra_sections.append(plan_section)

    prompt = build_prompt(
        question,
        file_path,
        file_error,
        extra_sections="\n\n".join(extra_sections) if extra_sections else None,
    )

    think_enabled = should_use_think_mode(question, file_path)

    traces: list[str] = []
    answers: list[str] = []
    runs = vote_runs()

    for run_index in range(runs):
        run_prompt = prompt
        if run_index > 0:
            run_prompt = f"{prompt}\n\nProvide an independent second attempt."
        raw, answer = run_with_self_correction(
            runner, run_prompt, question, file_path=file_path
        )
        result = verify_answer(question, raw)
        if not result.approved:
            retry_prompt = (
                f"{prompt}\n\nVerifier issues: {'; '.join(result.issues)}\n"
                f"Previous answer: {result.answer}\n"
                "Retry once with corrected final_answer."
            )
            retry_issues = _double_compute_hint(question, result.answer)
            if retry_issues:
                retry_prompt += "\n" + retry_issues[0]
            retry_raw = runner.run(
                retry_prompt,
                question=question,
                file_path=file_path,
                think=True,
            )
            retry_result = verify_answer(question, retry_raw)
            if retry_result.approved or len(retry_result.issues) <= len(result.issues):
                raw, answer = retry_raw, retry_result.answer
        traces.append(raw)
        answers.append(answer)

    final_answer = majority_vote(answers)
    combined_trace = traces[0] if len(traces) == 1 else "\n\n---VOTE---\n\n".join(traces)

    write_notes(
        task_id,
        f"# Notes\n\nStrategy: {plan.get('strategy', 'direct')}\n\nAnswer: {final_answer}\n",
    )
    write_evidence(
        task_id,
        {
            "question": question,
            "answer": final_answer,
            "depth": pipeline_depth(),
            "planned": should_plan(),
            "vote_runs": runs,
            "think_mode": think_enabled,
        },
    )
    return final_answer, combined_trace
