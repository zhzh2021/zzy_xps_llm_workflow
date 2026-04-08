"""Stable window detector for operando XAS."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def detect_stable_windows(df, feature_col: str, window_size: int = 5,
                          max_abs_slope: float = 0.001, x_col: Optional[str] = None) -> List[Dict[str, Any]]:
    if not HAS_PANDAS:
        raise ImportError("pandas is required for detect_stable_windows")
    if feature_col not in df.columns:
        return []

    x = np.asarray(df[x_col], dtype=float) if x_col and x_col in df.columns else np.arange(len(df))
    y = np.asarray(df[feature_col], dtype=float)

    stable = []
    n = len(y)
    for i in range(0, n - window_size + 1):
        xw = x[i:i + window_size]
        yw = y[i:i + window_size]
        if np.any(~np.isfinite(yw)):
            continue
        coeffs = np.polyfit(xw, yw, 1)
        slope = float(coeffs[0])
        if abs(slope) <= max_abs_slope:
            stable.append({"start": i, "end": i + window_size - 1, "slope": slope})

    return stable
