"""Public GAIA agent facade."""

from __future__ import annotations

from agent.agent_runner import AgentRunner
from agent.pipeline import run_pipeline


class GaiaAgent:
    def __init__(self):
        self.runner = AgentRunner()

    def __call__(
        self,
        question: str,
        file_path: str | None = None,
        file_error: str | None = None,
        task_id: str | None = None,
    ) -> str:
        answer, _trace = run_pipeline(
            self.runner,
            question,
            file_path=file_path,
            file_error=file_error,
            task_id=task_id,
        )
        print(f"Agent answer: {answer[:120]}")
        return answer
