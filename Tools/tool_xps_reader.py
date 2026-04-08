"""Wrapper for running the legacy XPS reader script via the chat tooling."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .tool_runner import run_python_script


SCRIPT_PATH = Path(__file__).resolve().parent / "XPS_reader" / "reader_main.py"


def run(project_root: Optional[str] = None, debug: bool = False, extra_args: Optional[Sequence[str]] = None):
    args = list(extra_args or [])
    if debug and "--debug" not in args:
        args.append("--debug")
    return run_python_script(
        SCRIPT_PATH,
        project_root=project_root,
        extra_args=args,
        friendly_name="XPS Reader",
    )


__all__ = ["run"]
