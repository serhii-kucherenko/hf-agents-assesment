"""Lightweight planner for standard/full pipeline depth."""

from __future__ import annotations

import json
import os
import re

from agent.artifacts.writer import write_plan
from model_provider import build_model


def pipeline_depth() -> str:
    return os.getenv("PIPELINE_DEPTH", "minimal").strip().lower()


def should_plan() -> bool:
    return pipeline_depth() in {"standard", "full"}


def _parse_plan_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def create_plan(
    question: str,
    file_hint: str | None,
    strategy_hints: str,
    task_id: str | None = None,
) -> dict:
    if not should_plan():
        return {"restatement": question, "strategy": "direct", "steps": []}

    model = build_model()
    prompt = (
        "Create a short JSON plan for this GAIA question.\n"
        'Return ONLY JSON: {"restatement": "...", "strategy": "...", '
        '"steps": [{"type": "lookup|compute|file|visual|synthesize", "goal": "..."}]}\n'
        "Keep 2-4 steps maximum.\n\n"
        f"Question: {question}\n"
    )
    if file_hint:
        prompt += f"\nAttachment: {file_hint}\n"
    if strategy_hints:
        prompt += f"\n{strategy_hints}\n"

    messages = [{"role": "user", "content": prompt}]
    response = model.generate(messages)
    raw = response.content if hasattr(response, "content") else str(response)
    plan = _parse_plan_json(str(raw)) or {
        "restatement": question,
        "strategy": "search and compute",
        "steps": [{"type": "synthesize", "goal": "Answer the question"}],
    }

    plan_md = (
        f"# Plan\n\n**Strategy:** {plan.get('strategy', '')}\n\n"
        f"**Restatement:** {plan.get('restatement', question)}\n\n"
        "## Steps\n"
    )
    for index, step in enumerate(plan.get("steps") or [], start=1):
        plan_md += f"{index}. [{step.get('type', '?')}] {step.get('goal', '')}\n"
    write_plan(task_id, plan_md)
    return plan


def plan_to_prompt_section(plan: dict) -> str:
    if not plan.get("steps"):
        return ""
    lines = ["Follow this plan:"]
    for index, step in enumerate(plan["steps"], start=1):
        lines.append(
            f"{index}. ({step.get('type', 'step')}) {step.get('goal', '')}"
        )
    return "\n".join(lines)
