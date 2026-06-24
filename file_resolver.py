"""Resolve GAIA task attachments from the local dataset or course API."""

from __future__ import annotations

from pathlib import Path

import requests

from gaia_data import resolve_task_file


def resolve_task_attachment(
    api_url: str,
    task_id: str,
    file_name: str,
    download_dir: Path,
) -> tuple[str | None, str | None]:
    local_path, local_error = resolve_task_file(task_id)
    if local_path:
        print(f"Using GAIA dataset file for task {task_id}: {local_path}")
        return local_path, None

    if not file_name:
        return None, local_error

    file_url = f"{api_url}/files/{task_id}"
    destination = download_dir / file_name
    print(f"GAIA dataset file unavailable, trying API: {file_url}")

    try:
        response = requests.get(file_url, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return str(destination.resolve()), None
    except requests.exceptions.HTTPError as error:
        message = f"{error.response.status_code} for {file_url}"
        if local_error:
            message = f"{local_error}; API fallback failed with {message}"
        return None, message
    except requests.exceptions.RequestException as error:
        message = str(error)
        if local_error:
            message = f"{local_error}; API fallback failed with {message}"
        return None, message
