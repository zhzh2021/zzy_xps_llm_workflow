"""Wrapper for the xps_workflow_runner script."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .tool_runner import run_python_script


SCRIPT_PATH = Path(__file__).resolve().parent / "xps_workflow_runner.py"


def run(project_root: Optional[str] = None, extra_args: Optional[Sequence[str]] = None):
    args = list(extra_args or [])
    if project_root:
        args.insert(0, str(project_root))
    return run_python_script(
        SCRIPT_PATH,
        project_root=project_root,
        extra_args=args,
        friendly_name="XPS Workflow Runner",
    )


__all__ = ["run"]
