"""Write plan.md, notes.md, and evidence.json artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path


def artifacts_enabled() -> bool:
    return os.getenv("AGENT_ARTIFACTS", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def artifact_dir(task_id: str | None) -> Path:
    base = Path(os.getenv("AGENT_ARTIFACTS_DIR", "artifacts"))
    name = task_id or "unknown"
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_plan(task_id: str | None, plan_md: str) -> None:
    if not artifacts_enabled():
        return
    (artifact_dir(task_id) / "plan.md").write_text(plan_md, encoding="utf-8")


def write_notes(task_id: str | None, notes_md: str) -> None:
    if not artifacts_enabled():
        return
    (artifact_dir(task_id) / "notes.md").write_text(notes_md, encoding="utf-8")


def write_evidence(task_id: str | None, evidence: dict) -> None:
    if not artifacts_enabled():
        return
    path = artifact_dir(task_id) / "evidence.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
