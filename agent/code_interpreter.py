"""Local Python/bash code execution for the GAIA agent."""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import tempfile
import traceback
import uuid
from typing import Any


class CodeInterpreter:
    def __init__(self, max_execution_time: int = 30):
        self.max_execution_time = max_execution_time
        self.globals: dict[str, Any] = {
            "__builtins__": __builtins__,
        }

    def execute_code(self, code: str, language: str = "python") -> dict[str, Any]:
        language = language.lower()
        execution_id = str(uuid.uuid4())
        if language == "python":
            return self._execute_python(code, execution_id)
        if language == "bash":
            return self._execute_bash(code, execution_id)
        return {
            "execution_id": execution_id,
            "status": "error",
            "stdout": "",
            "stderr": f"Unsupported language: {language}",
            "result": None,
        }

    def _execute_python(self, code: str, execution_id: str) -> dict[str, Any]:
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        result: dict[str, Any] = {
            "execution_id": execution_id,
            "status": "error",
            "stdout": "",
            "stderr": "",
            "result": None,
        }
        try:
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(
                error_buffer
            ):
                exec(code, self.globals)
            result["status"] = "success"
            result["stdout"] = output_buffer.getvalue()
            result["stderr"] = error_buffer.getvalue()
        except Exception:
            result["stderr"] = f"{error_buffer.getvalue()}\n{traceback.format_exc()}"
        return result

    def _execute_bash(self, code: str, execution_id: str) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.max_execution_time,
            )
            return {
                "execution_id": execution_id,
                "status": "success" if completed.returncode == 0 else "error",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "result": None,
            }
        except subprocess.TimeoutExpired:
            return {
                "execution_id": execution_id,
                "status": "error",
                "stdout": "",
                "stderr": "Execution timed out.",
                "result": None,
            }


_interpreter = CodeInterpreter()


def run_python(code: str) -> str:
    result = _interpreter.execute_code(code, language="python")
    parts: list[str] = []
    if result["status"] == "success":
        parts.append("Code executed successfully.")
        if result.get("stdout"):
            parts.append(f"Output:\n{result['stdout'].strip()}")
    else:
        parts.append("Code execution failed.")
        if result.get("stderr"):
            parts.append(f"Error:\n{result['stderr'].strip()}")
    return "\n".join(parts)[:12000]
