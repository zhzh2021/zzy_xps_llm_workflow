"""Wrapper for the XPS quantification script."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .tool_runner import run_python_script


SCRIPT_PATH = Path(__file__).resolve().parent / "XPS_Quantifier" / "XPS_Quantifier.py"


def run(project_root: Optional[str] = None, extra_args: Optional[Sequence[str]] = None):
    return run_python_script(
        SCRIPT_PATH,
        project_root=project_root,
        extra_args=extra_args,
        friendly_name="XPS Quantifier",
    )


__all__ = ["run"]
