"""Wrapper exposing the workflow entrypoint as `full_analysis` for the LLM.

This simply delegates to the real workflow implementation so the assistant
can call `full_analysis` (as requested by the system prompt) and run the
production workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .tool_runner import run_python_script


SCRIPT_PATH = Path(__file__).resolve().parent / "real_xps_workflow.py"


def run(project_root: Optional[str] = None, extra_args: Optional[Sequence[str]] = None):
    args = list(extra_args or [])
    if project_root:
        args.insert(0, str(project_root))
    return run_python_script(
        SCRIPT_PATH,
        project_root=project_root,
        extra_args=args,
        friendly_name="Full Analysis (real_xps_workflow)",
    )


__all__ = ["run"]
