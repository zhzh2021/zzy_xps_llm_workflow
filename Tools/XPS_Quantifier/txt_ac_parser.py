"""
Optional TXT Atomic Concentration parser for XPS.

This module is intentionally separate so projects can opt-in to
TXT-based extraction. The main workflow defaults to JSON/RSF
based computation.
"""
from __future__ import annotations
from pathlib import Path
import re
from typing import Dict, List, Optional

# Local regex patterns for header/rows detection
ELEMENT_TOKEN_RE = re.compile(r"[A-Za-z]{1,2}\d+[spdfSPDF]\d*")
MEAN_RE = re.compile(r"\bMean\b", re.I)
STD_RE = re.compile(r"(Standard\s+Deviation|Std\.?\s*Dev\.?)", re.I)
RSF_RE = re.compile(r"\b(RSF|CorrectedRSF)\b", re.I)
SECTION_AC_RE = re.compile(r"\bAtomic Concentration Table\b", re.I)
SECTION_NEXT_RE = re.compile(r"\bWeight % Table\b", re.I)


def _is_dash_line(s: str) -> bool:
    return set(s.strip()) <= set("- ")


def _find_atomic_concentration_section(lines: List[str]) -> Optional[int]:
    for i, l in enumerate(lines):
        if SECTION_AC_RE.search(l):
            return i
    return None


def _detect_header(lines: List[str], start_idx: int) -> Optional[Dict]:
    for j in range(start_idx + 1, len(lines)):
        s = lines[j].strip()
        if not s or _is_dash_line(s):
            continue
        tokens = s.split()
        if tokens and all(ELEMENT_TOKEN_RE.fullmatch(t) for t in tokens):
            return {"header": tokens, "header_idx": j}
        if len(tokens) >= 2 and all(any(c.isalpha() for c in t) and any(c.isdigit() for c in t) for t in tokens):
            return {"header": tokens, "header_idx": j}
    return None


def parse_atomic_concentration_rows(file_path: Path, cfg) -> Optional[Dict]:
    """Parse a single TXT file to extract atomic concentration data.

    Returns dict with keys: sample, header, rows; or None if not found.
    """
    text = file_path.read_text(errors="ignore")
    lines = text.splitlines()

    # Extract sample name
    sample = None
    for line in lines:
        if line.strip().startswith("File Name"):
            sample = line.split(":", 1)[1].strip().strip('"')
            break
    if not sample:
        sample = file_path.stem

    sec_idx = _find_atomic_concentration_section(lines)
    if sec_idx is None:
        return None

    hdr_info = _detect_header(lines, sec_idx)
    if hdr_info is None:
        return None
    header = hdr_info["header"]
    header_idx = hdr_info["header_idx"]

    rows: List[List[float]] = []
    for k in range(header_idx + 1, len(lines)):
        s = lines[k].strip()
        if not s:
            continue
        if SECTION_NEXT_RE.search(s):
            break
        if _is_dash_line(s):
            continue
        if MEAN_RE.search(s) or STD_RE.search(s):
            if getattr(cfg, "parsing", None) and not getattr(cfg.parsing, "include_std_rows", False):
                continue
        if getattr(cfg, "parsing", None) and getattr(cfg.parsing, "skip_rsf_rows", True) and RSF_RE.search(s):
            continue

        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", s)
        if len(nums) >= len(header):
            vals = list(map(float, nums[: len(header)]))
            rows.append(vals)

    if not rows:
        return None

    return {"sample": sample, "header": header, "rows": rows}

