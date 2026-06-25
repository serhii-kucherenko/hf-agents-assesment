"""GAIA dataset access disabled — use course API and local fixtures only."""

from __future__ import annotations

_DISABLED_MSG = (
    "Direct GAIA benchmark dataset download is disabled. "
    "Use the course scoring API (eval/course_client.py) or eval/fixtures/ for evaluation."
)


def get_gaia_data_dir() -> str:
    raise RuntimeError(_DISABLED_MSG)


def get_gaia_index() -> dict[str, dict]:
    raise RuntimeError(_DISABLED_MSG)


def resolve_task_file(task_id: str) -> tuple[str | None, str | None]:
    return None, _DISABLED_MSG


def get_task_metadata(task_id: str) -> dict | None:
    return None


def get_ground_truth(task_id: str) -> str | None:
    return None
