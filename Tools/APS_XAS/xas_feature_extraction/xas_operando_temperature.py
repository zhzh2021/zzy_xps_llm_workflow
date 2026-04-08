"""Temperature-correlated feature change for operando XAS."""
from __future__ import annotations
from typing import Dict, Any
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    pearsonr = None
    HAS_SCIPY = False


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return {"r": np.nan, "p": np.nan, "n": int(x.size)}
    if HAS_SCIPY:
        r, p = pearsonr(x, y)
        return {"r": float(r), "p": float(p), "n": int(x.size)}
    r = np.corrcoef(x, y)[0, 1]
    return {"r": float(r), "p": np.nan, "n": int(x.size)}


def temperature_correlation(df, feature_col: str, temp_col: str = "temperature") -> Dict[str, Any]:
    if not HAS_PANDAS:
        raise ImportError("pandas is required for temperature_correlation")
    if feature_col not in df.columns or temp_col not in df.columns:
        return {"r": np.nan, "p": np.nan, "n": 0}
    return _safe_corr(np.asarray(df[feature_col], dtype=float), np.asarray(df[temp_col], dtype=float))
