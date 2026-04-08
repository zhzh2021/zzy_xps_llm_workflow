"""Wrapper that exposes the smart grouping helper as a run() target."""

from __future__ import annotations

import json
from typing import Sequence

from .create_groups import smart_define_groups


def run(samples: Sequence[str], instructions: str):
    """Execute the smart grouping helper and format the response."""
    if hasattr(smart_define_groups, "invoke"):
        payload = smart_define_groups.invoke({
            "samples": list(samples),
            "instructions": instructions,
        })
    else:
        payload = smart_define_groups(samples=list(samples), instructions=instructions)

    message = payload.get("message", "Groups created")
    groups = payload.get("groups", {})
    formatted = json.dumps(groups, indent=2)
    return True, f"{message}\n{formatted}"


__all__ = ["run"]
