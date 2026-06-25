"""Answer extraction and format verification."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import requests

from agent.planner import pipeline_depth
from model_provider import build_verifier_model
from scoring import normalize_answer


@dataclass
class VerifierResult:
    approved: bool
    answer: str
    issues: list[str]


def strip_reasoning_blocks(text: str) -> str:
    cleaned = text
    for tag in ("redacted_reasoning", "think", "thinking"):
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        pattern = re.escape(open_tag) + r".*?" + re.escape(close_tag)
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def extract_answer(raw_result: str) -> str:
    text = strip_reasoning_blocks(str(raw_result).strip())

    final_match = re.search(
        r'final_answer\s*\(\s*["\'](.+?)["\']\s*\)',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if final_match:
        return normalize_answer(final_match.group(1))

    final_answer_match = re.search(
        r"FINAL ANSWER:\s*(.+?)(?:\n|$)",
        text,
        flags=re.IGNORECASE,
    )
    if final_answer_match:
        return normalize_answer(final_answer_match.group(1))

    if len(text) < 200 and "def " not in text and "import " not in text:
        return normalize_answer(text)

    if "```" in text:
        code_blocks = re.findall(r"```(?:\w*\n)?(.*?)```", text, flags=re.DOTALL)
        for block in reversed(code_blocks):
            block_match = re.search(
                r'final_answer\s*\(\s*["\'](.+?)["\']\s*\)',
                block,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if block_match:
                return normalize_answer(block_match.group(1))

    return normalize_answer(text)


def _format_issues(answer: str) -> list[str]:
    issues: list[str] = []
    if not answer or not answer.strip():
        issues.append("Answer is empty.")
    if "\n" in answer.strip():
        issues.append("Answer contains newlines; GAIA expects a single line value.")
    if len(answer) > 500:
        issues.append("Answer is unusually long for GAIA formatting.")
    return issues


def _spot_check_urls(raw_trace: str) -> list[str]:
    issues: list[str] = []
    urls = re.findall(r"https?://[^\s\)\]\"']+", raw_trace)
    for url in urls[:2]:
        try:
            response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if response.status_code >= 400:
                issues.append(f"Source URL not reachable: {url}")
        except requests.RequestException:
            issues.append(f"Could not fetch cited URL: {url}")
    return issues


def _critic_check(question: str, answer: str, trace: str) -> list[str]:
    if pipeline_depth() not in {"standard", "full"}:
        return []
    if not os.getenv("CRITIC_MODEL") and pipeline_depth() == "standard":
        return []

    model = build_verifier_model()
    prompt = (
        "Does the evidence support this GAIA answer? Reply YES or NO then one sentence why.\n"
        f"Question: {question}\nAnswer: {answer}\nEvidence excerpt:\n{trace[:3000]}"
    )
    try:
        response = model.generate([{"role": "user", "content": prompt}])
        text = (response.content if hasattr(response, "content") else str(response)).strip()
        if text.upper().startswith("NO"):
            return [text]
    except Exception as error:
        return [f"Critic check skipped: {error}"]
    return []


def verify_answer(question: str, raw_result: str) -> VerifierResult:
    answer = extract_answer(raw_result)
    issues = _format_issues(answer)
    issues.extend(_spot_check_urls(str(raw_result)))
    issues.extend(_critic_check(question, answer, str(raw_result)))
    return VerifierResult(approved=len(issues) == 0, answer=answer, issues=issues)
