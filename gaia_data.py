"""Load GAIA benchmark files locally from the official Hugging Face dataset."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

GAIA_REPO = "gaia-benchmark/GAIA"
GAIA_CONFIG = os.getenv("GAIA_CONFIG", "2023_level1")
GAIA_SPLIT = os.getenv("GAIA_SPLIT", "validation")


@lru_cache(maxsize=1)
def get_gaia_data_dir() -> str:
    configured = os.getenv("GAIA_DATA_DIR", "").strip()
    if configured:
        return configured
    return snapshot_download(repo_id=GAIA_REPO, repo_type="dataset")


@lru_cache(maxsize=1)
def get_gaia_index() -> dict[str, dict]:
    data_dir = get_gaia_data_dir()
    dataset = load_dataset(data_dir, GAIA_CONFIG, split=GAIA_SPLIT)
    return {row["task_id"]: dict(row) for row in dataset}


def resolve_task_file(task_id: str) -> tuple[str | None, str | None]:
    """Return a local file path for a GAIA task, if the dataset includes one."""
    row = get_gaia_index().get(task_id)
    if row is None:
        return None, f"task_id {task_id} not found in GAIA {GAIA_SPLIT} split"

    file_path = (row.get("file_path") or "").strip()
    if not file_path:
        return None, None

    full_path = Path(get_gaia_data_dir()) / file_path
    if not full_path.exists():
        return None, f"GAIA file missing on disk: {full_path}"

    return str(full_path.resolve()), None


def get_task_metadata(task_id: str) -> dict | None:
    return get_gaia_index().get(task_id)
