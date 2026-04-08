"""Utility helpers for invoking the legacy XPS scripts as subprocesses."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


DEFAULT_TIMEOUT_SECONDS = 60 * 60  # 1 hour


def _normalize_args(extra_args: Optional[Sequence[object]]) -> list[str]:
    if not extra_args:
        return []
    normalized: list[str] = []
    for item in extra_args:
        if item is None:
            continue
        normalized.append(str(item))
    return normalized


def run_python_script(
    script_path: Path,
    *,
    project_root: Optional[str] = None,
    extra_args: Optional[Sequence[object]] = None,
    friendly_name: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Tuple[bool, str]:
    """Execute the given Python script in a subprocess and capture output."""
    script_path = Path(script_path).resolve()
    if not script_path.exists():
        return False, f"Script not found: {script_path}"

    cmd = [sys.executable, str(script_path)]
    cmd.extend(_normalize_args(extra_args))

    env = os.environ.copy()
    if project_root:
        env["XPS_PROJECT_ROOT"] = str(Path(project_root).resolve())

    try:
        result = subprocess.run(
            cmd,
            cwd=str(script_path.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=timeout or DEFAULT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        label = friendly_name or script_path.name
        return False, f"{label} timed out after {timeout or DEFAULT_TIMEOUT_SECONDS}s."
    except Exception as exc:  # pragma: no cover - defensive
        label = friendly_name or script_path.name
        return False, f"Failed to launch {label}: {exc}"

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    label = friendly_name or script_path.name

    if result.returncode == 0:
        message_parts: list[str] = []
        if stdout:
            message_parts.append(stdout)
        else:
            message_parts.append(f"{label} completed successfully.")

        if stderr:
            message_parts.append("[stderr]")
            message_parts.append(stderr)

        return True, "\n".join(message_parts).strip()

    error_msg = [f"{label} failed with exit code {result.returncode}."]
    if stderr:
        error_msg.append("Error output:")
        error_msg.append(stderr)
    elif stdout:
        error_msg.append("Process output:")
        error_msg.append(stdout)
    return False, "\n".join(error_msg).strip()


__all__ = ["run_python_script"]
