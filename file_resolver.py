"""Resolve GAIA task attachments from the course scoring API."""

from __future__ import annotations

from pathlib import Path

import requests


def resolve_task_attachment(
    api_url: str,
    task_id: str,
    file_name: str,
    download_dir: Path,
) -> tuple[str | None, str | None]:
    if not file_name:
        return None, None

    file_url = f"{api_url.rstrip('/')}/files/{task_id}"
    destination = download_dir / file_name
    print(f"Downloading attachment from course API: {file_url}")

    try:
        response = requests.get(file_url, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return str(destination.resolve()), None
    except requests.exceptions.HTTPError as error:
        message = f"{error.response.status_code} for {file_url}"
        return None, message
    except requests.exceptions.RequestException as error:
        return None, str(error)
