from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

# Reuse discovery from Tools-backed utils
from zzy_llm.Tools.utils import ExperimentSpec, build_manifest

DEFAULT_MANIFEST_FILENAME = "_manifest.json"


def export_manifest_json() -> Path:
    """
    Export live manifest (discovered via zzy_llm.Tools.exp_*) to
    zzy_llm/experiments/_manifest.json for packaging.
    """
    pkg_dir = Path(__file__).parent
    manifest = build_manifest()  # uses Tools-backed discovery
    data = [m.to_dict() for m in manifest]

    pkg_dir.mkdir(parents=True, exist_ok=True)
    path = pkg_dir / DEFAULT_MANIFEST_FILENAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_manifest_from_json() -> List[ExperimentSpec]:
    """Load manifest JSON from zzy_llm/experiments/_manifest.json if available."""
    path = Path(__file__).parent / DEFAULT_MANIFEST_FILENAME
    try:
        arr = json.loads(path.read_text(encoding="utf-8"))
        # reconstruct ExperimentSpec objects
        return [ExperimentSpec.from_dict(x) for x in arr]  # type: ignore[attr-defined]
    except Exception:
        return []

