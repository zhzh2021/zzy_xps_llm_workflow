"""Significant change detector for operando XAS."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def detect_significant_changes(df, feature_col: str, zscore_thresh: float = 2.0,
                               min_delta: Optional[float] = None, window: int = 3) -> List[Dict[str, Any]]:
    if not HAS_PANDAS:
        raise ImportError("pandas is required for detect_significant_changes")
    if feature_col not in df.columns:
        return []

    y = np.asarray(df[feature_col], dtype=float)
    if len(y) < window + 1:
        return []

    baseline = np.convolve(y, np.ones(window) / window, mode="valid")
    baseline = np.concatenate([np.full(window - 1, baseline[0]), baseline])
    diff = y - baseline

    mu = np.nanmean(diff)
    sigma = np.nanstd(diff) + 1e-12
    z = (diff - mu) / sigma

    changes = []
    for i in range(len(y)):
        if abs(z[i]) >= zscore_thresh:
            if min_delta is not None and abs(diff[i]) < min_delta:
                continue
            changes.append({"index": i, "delta": float(diff[i]), "z": float(z[i])})

    return changes
