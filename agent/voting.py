"""Pass@N majority voting for full pipeline depth."""

from __future__ import annotations

import os
from collections import Counter

from agent.planner import pipeline_depth


def vote_runs() -> int:
    if pipeline_depth() != "full":
        return 1
    return max(1, int(os.getenv("VOTE_RUNS", "1")))


def majority_vote(answers: list[str]) -> str:
    if not answers:
        return ""
    if len(answers) == 1:
        return answers[0]
    normalized = [answer.strip().lower() for answer in answers if answer.strip()]
    if not normalized:
        return answers[0]
    winner, _ = Counter(normalized).most_common(1)[0]
    for answer in answers:
        if answer.strip().lower() == winner:
            return answer
    return answers[0]
