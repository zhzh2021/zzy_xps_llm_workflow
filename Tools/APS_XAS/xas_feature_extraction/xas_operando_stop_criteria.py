"""Stop criteria for operando XAS."""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def stop_criteria(df, feature_col: str, window_size: int = 5, max_range: float = 0.002,
                  snr_col: Optional[str] = None, min_snr: Optional[float] = None) -> Dict[str, Any]:
    if not HAS_PANDAS:
        raise ImportError("pandas is required for stop_criteria")
    if feature_col not in df.columns:
        return {"stop": False, "reasons": ["feature_missing"]}

    if len(df) < window_size:
        return {"stop": False, "reasons": ["insufficient_points"]}

    y = np.asarray(df[feature_col], dtype=float)
    window = y[-window_size:]
    if np.any(~np.isfinite(window)):
        return {"stop": False, "reasons": ["invalid_values"]}

    reasons = []
    if float(np.nanmax(window) - np.nanmin(window)) <= max_range:
        reasons.append("feature_stable")

    if snr_col and min_snr is not None and snr_col in df.columns:
        snr = float(df[snr_col].iloc[-1])
        if snr < min_snr:
            reasons.append("snr_below_min")

    return {"stop": len(reasons) > 0, "reasons": reasons}
