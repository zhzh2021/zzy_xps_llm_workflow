"""Discovery helpers shared by the chat UI and workflow router."""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MANIFEST_FILENAME = "_manifest.json"


@dataclass
class ExperimentSpec:
    """Structured metadata describing a callable experiment/tool."""

    name: str
    module: str
    doc: str = ""
    args_required: List[str] = field(default_factory=list)
    args_optional: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentSpec":
        return cls(
            name=payload.get("name", ""),
            module=payload.get("module", ""),
            doc=payload.get("doc", ""),
            args_required=list(payload.get("args_required", []) or []),
            args_optional=list(payload.get("args_optional", []) or []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _manifest_path() -> Path:
    """
    Resolve the manifest path even when the app is relocated (e.g., PyInstaller).

    Order of checks:
      1. Same directory as this module.
      2. ZZY_LLM_HOME env var (supports pointing to repo or packaged root).
      3. PyInstaller's _MEIPASS extraction directory.
    """
    here = Path(__file__).resolve()
    candidates: List[Path] = [here.with_name(MANIFEST_FILENAME)]

    env_home = os.environ.get("ZZY_LLM_HOME")
    if env_home:
        base = Path(env_home)
        candidates.extend([
            base / "Tools" / MANIFEST_FILENAME,
            base / "zzy_llm" / "Tools" / MANIFEST_FILENAME,
        ])

    meipass = getattr(sys, "_MEIPASS", None)  # type: ignore[attr-defined]
    if meipass:
        base = Path(meipass)
        candidates.extend([
            base / "Tools" / MANIFEST_FILENAME,
            base / "zzy_llm" / "Tools" / MANIFEST_FILENAME,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_manifest() -> List[ExperimentSpec]:
    """Load manifest specs from Tools/_manifest.json."""
    manifest_file = _manifest_path()
    if not manifest_file.exists():
        return []

    try:
        raw_text = manifest_file.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return []

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return []

    manifest: List[ExperimentSpec] = []
    for entry in data:
        try:
            spec = ExperimentSpec.from_dict(entry)
            if spec.name and spec.module:
                manifest.append(spec)
        except Exception:
            continue
    return manifest


def _validate_args(spec: ExperimentSpec, args: Dict[str, Any]) -> Optional[str]:
    missing = [key for key in spec.args_required if key not in args]
    if missing:
        return f"Missing required args for {spec.name}: {', '.join(missing)}"
    return None


def _coerce_result(spec_name: str, result: Any) -> Tuple[bool, str]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool):
        success, message = result
        return success, str(message)
    if isinstance(result, str):
        return True, result
    if result is None:
        return True, f"{spec_name} completed."
    return True, json.dumps(result, default=str)


def run_experiment(name: str, args: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Import the module specified in the manifest and invoke its run() helper."""
    args = args or {}
    manifest = build_manifest()
    spec = next((m for m in manifest if m.name == name), None)
    if not spec:
        return False, f"Experiment '{name}' not found in manifest."

    error = _validate_args(spec, args)
    if error:
        return False, error

    try:
        module = importlib.import_module(spec.module)
    except ModuleNotFoundError as exc:
        return False, f"Cannot import module '{spec.module}': {exc}"
    except Exception as exc:  # pragma: no cover
        return False, f"Import failed for '{spec.module}': {exc}"

    runner = getattr(module, "run", None)
    if runner is None or not callable(runner):
        # Allow modules that expose a callable named `main` that accepts kwargs
        runner = getattr(module, "main", None)
        if runner is None or not callable(runner):
            return False, f"Module '{spec.module}' does not export a callable run()."

    try:
        result = runner(**args)
    except TypeError as exc:
        return False, f"Argument mismatch for {spec.name}: {exc}"
    except Exception as exc:
        tb = traceback.format_exc()
        return False, f"{spec.name} raised an exception: {exc}\n{tb}"

    return _coerce_result(spec.name, result)


__all__ = ["ExperimentSpec", "build_manifest", "run_experiment"]
