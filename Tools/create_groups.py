"""generic group tools for files based on rules:
pattern matching
index ranges
metadata filters
tags
regex
numeric ranges
"""
# smart_grouping.py

import re
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Union
from langchain_core.tools import tool

GROUP_MEMORY_FILE = "groups_memory.json"
GROUP_WORKSPACE_ROOT = "_group_runs"
REQUIRED_WORKFLOW_DIRS = [
    "00_raw_data",
    "01_converted_csv",
    "02_fitted_results",
    "03_quantified_data",
    "04_plots",
    "05_correlator_results",
    "_logs",
    "_temp",
]


def group_files_by_prefix(raw_dir: Path) -> Dict[str, List[Path]]:
    """Group raw data files by prefix (segment before '_' or '-')."""
    groups: Dict[str, List[Path]] = {}
    for file in raw_dir.iterdir():
        if not file.is_file():
            continue
        stem = file.stem
        if "_" in stem:
            prefix = stem.split("_", 1)[0]
        elif "-" in stem:
            prefix = stem.split("-", 1)[0]
        else:
            match = re.match(r"([A-Za-z0-9]+)", stem)
            prefix = match.group(1) if match else stem
        key = prefix.lower()
        groups.setdefault(key, []).append(file)
    return groups


def prepare_group_workspace(base_project: Path, group_name: str, files: List[Path]) -> Path:
    """Create a temporary workflow workspace that only contains the group's files."""
    workspace_root = base_project / GROUP_WORKSPACE_ROOT / group_name
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    for dirname in REQUIRED_WORKFLOW_DIRS:
        (workspace_root / dirname).mkdir(parents=True, exist_ok=True)

    src_config = base_project / "xps_config"
    dest_config = workspace_root / "xps_config"
    if src_config.exists():
        shutil.copytree(src_config, dest_config, dirs_exist_ok=True)
    else:
        dest_config.mkdir(parents=True, exist_ok=True)

    raw_target = workspace_root / "00_raw_data"
    for file in files:
        shutil.copy2(file, raw_target / file.name)

    return workspace_root


# -------------------------------
# Memory I/O
# -------------------------------

def load_group_memory() -> Dict:
    if os.path.exists(GROUP_MEMORY_FILE):
        with open(GROUP_MEMORY_FILE) as f:
            return json.load(f)
    return {}


def save_group_memory(groups: Dict):
    with open(GROUP_MEMORY_FILE, "w") as f:
        json.dump(groups, f, indent=2)


# -------------------------------
# Helper Functions
# -------------------------------

def normalize(x):
    """Normalize sample label."""
    return str(x).strip()


def normalize_list(lst):
    return [normalize(i) for i in lst]


def expand_range(a, b):
    """Expand numeric ranges: 1–12 → ['1', ..., '12']."""
    try:
        start, end = int(a), int(b)
        return [str(i) for i in range(start, end + 1)]
    except:
        return [str(a), str(b)]


def extract_numbers(name: str):
    """Extract digits from sample labels like S12 → 12."""
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


# -------------------------------
# Main Smart Grouping Capability
# -------------------------------

@tool
def smart_define_groups(
    samples: List[Union[int, str]],
    instructions: str
) -> Dict:
    """
    Smart grouping tool. Understands natural language instructions like:
      - "Group 1–12 as A and 13–20 as B"
      - "Group first half as A, second half as B"
      - "Group odd vs even"
      - "Group every 5 samples"
      - "Group by prefix Fe*, Cu*"

    Automatically infers rules and builds consistent groups.
    """

    samples = normalize_list(samples)
    groups = {}

    n = len(samples)
    numeric_samples = [extract_numbers(s) for s in samples]
    numeric_available = all(x is not None for x in numeric_samples)

    text = instructions.lower().strip()

    # ------------------------------------------------------
    # RULE 1 — Explicit ranges: "1-12 as A"
    # ------------------------------------------------------
    range_pattern = r"(\d+)\s*[-–]\s*(\d+)\s*(?:as|=)\s*([a-zA-Z0-9_]+)"
    matches = re.findall(range_pattern, text)
    if matches:
        for start, end, gname in matches:
            groups[gname] = expand_range(start, end)

        save_group_memory(groups)
        return {"message": "Range-based groups created", "groups": groups}

    # ------------------------------------------------------
    # RULE 2 — Odd vs even
    # ------------------------------------------------------
    if "odd" in text or "even" in text:
        if not numeric_available:
            raise ValueError("Odd/even grouping requires numeric sample IDs.")

        odds = []
        evens = []
        for s, nval in zip(samples, numeric_samples):
            (odds if nval % 2 == 1 else evens).append(s)

        if "odd" in text:
            groups["odd"] = odds
        if "even" in text:
            groups["even"] = evens

        save_group_memory(groups)
        return {"message": "Odd/even groups created", "groups": groups}

    # ------------------------------------------------------
    # RULE 3 — First/second/third segments
    # “first 12 as A”, “remaining as B”
    # ------------------------------------------------------
    seg_pattern = r"first\s+(\d+)\s+(?:samples?)\s*(?:as|=)\s*([a-zA-Z0-9_]+)"
    seg_match = re.search(seg_pattern, text)
    if seg_match:
        k = int(seg_match.group(1))
        gname = seg_match.group(2)
        groups[gname] = samples[:k]

        # find “rest as B”
        rest_match = re.search(
            r"(rest|remaining)\s*(?:as|=)\s*([a-zA-Z0-9_]+)", text)
        if rest_match:
            rest_group_name = rest_match.group(2)
            groups[rest_group_name] = samples[k:]

        save_group_memory(groups)
        return {"message": "Segment-based groups created", "groups": groups}

    # ------------------------------------------------------
    # RULE 4 — Group every N samples
    # e.g., “group every 5 samples”
    # ------------------------------------------------------
    every_pattern = r"every\s+(\d+)\s+(?:samples?)"
    em = re.search(every_pattern, text)
    if em:
        step = int(em.group(1))
        group_index = 1
        for i in range(0, n, step):
            groups[f"G{group_index}"] = samples[i: i + step]
            group_index += 1

        save_group_memory(groups)
        return {"message": "Step-based groups created", "groups": groups}

    # ------------------------------------------------------
    # RULE 5 — Prefix-based grouping (Fe1, Fe2 vs Cu1, Cu2)
    # ------------------------------------------------------
    if "prefix" in text or "by prefix" in text:
        prefixes = {}
        for s in samples:
            m = re.match(r"([A-Za-z]+)", s)
            prefix = m.group(1).lower() if m else "other"
            prefixes.setdefault(prefix, []).append(s)

        groups = prefixes
        save_group_memory(groups)
        return {"message": "Prefix-based groups created", "groups": groups}

    # ------------------------------------------------------
    # RULE 6 — Explicit naming inside instructions
    # "A = 1,2,3 and B = 4,5,6"
    # ------------------------------------------------------
    explicit_pattern = r"([A-Za-z0-9_]+)\s*=\s*([\d,\s]+)"
    matches = re.findall(explicit_pattern, text)
    if matches:
        for gname, nums in matches:
            parsed = [normalize(x) for x in re.split(r"[,\s]+", nums) if x]
            groups[gname] = parsed

        save_group_memory(groups)
        return {"message": "Explicit groups created", "groups": groups}

    # ------------------------------------------------------
    # If no rule matched → raise friendly error
    # ------------------------------------------------------
    raise ValueError(
        "Could not interpret grouping instructions. "
        "Try formats like:\n"
        " - 'Group 1–12 as A and 13–20 as B'\n"
        " - 'Group odd vs even'\n"
        " - 'Group first 12 samples as A, rest as B'\n"
        " - 'Group every 5 samples'\n"
        " - 'Group by prefix'\n"
    )


@tool
def run_grouped_workflow_by_prefix(
    project_root: str,
    min_files_per_group: int = 1,
) -> Dict:
    """
    Group files located in project_root/00_raw_data by prefix and run the full workflow for each group.

    Args:
        project_root: Path to the base project containing the standard workflow folders.
        min_files_per_group: Minimum number of files required to run a group workflow.

    Returns:
        Dict containing group assignments, workflow statuses, and stacked plot destinations.
    """

    project_root_path = Path(project_root).resolve()
    raw_dir = project_root_path / "00_raw_data"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    grouped = group_files_by_prefix(raw_dir)
    grouped = {k: v for k, v in grouped.items() if len(v) >= min_files_per_group}
    if not grouped:
        raise ValueError("No groups satisfied the minimum file count requirement.")

    workspace_root = project_root_path / GROUP_WORKSPACE_ROOT
    workspace_root.mkdir(parents=True, exist_ok=True)

    runner_script = Path(__file__).resolve().parent / "real_xps_workflow.py"
    if not runner_script.exists():
        raise FileNotFoundError(f"real_xps_workflow.py not found at {runner_script}")

    summary = {}
    for prefix, files in grouped.items():
        workspace = prepare_group_workspace(project_root_path, prefix, files)

        result = subprocess.run(
            [sys.executable, str(runner_script), str(workspace)],
            capture_output=True,
            text=True,
            cwd=str(workspace),
        )

        stacked_src = workspace / "04_plots" / "02_peak_fitting" / "stacked_comparison"
        stacked_dest = project_root_path / "04_plots" / "02_peak_fitting" / "group_comparisons" / prefix
        if stacked_dest.exists():
            shutil.rmtree(stacked_dest)
        if stacked_src.exists():
            stacked_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(stacked_src, stacked_dest)
        else:
            stacked_dest = None

        summary[prefix] = {
            "files": [f.name for f in files],
            "workflow_status": "success" if result.returncode == 0 else "failed",
            "stacked_plots_dir": str(stacked_dest) if stacked_dest else None,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }

    save_group_memory({k: [f.name for f in v] for k, v in grouped.items()})

    return {
        "message": f"Grouped workflow completed for {len(summary)} groups",
        "groups": summary,
        "workspace_root": str(workspace_root),
    }
