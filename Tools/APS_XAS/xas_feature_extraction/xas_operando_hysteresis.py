"""Hysteresis indicator for operando XAS."""
from __future__ import annotations
from typing import Dict, Any
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def hysteresis_indicator(df, feature_col: str, condition_col: str = "ramp_direction",
                         up_value: str = "up", down_value: str = "down", normalize: bool = True) -> Dict[str, Any]:
    if not HAS_PANDAS:
        raise ImportError("pandas is required for hysteresis_indicator")
    if feature_col not in df.columns or condition_col not in df.columns:
        return {"delta": np.nan, "mean_up": np.nan, "mean_down": np.nan}

    up = df[df[condition_col] == up_value][feature_col]
    down = df[df[condition_col] == down_value][feature_col]
    if len(up) == 0 or len(down) == 0:
        return {"delta": np.nan, "mean_up": np.nan, "mean_down": np.nan}

    mean_up = float(np.nanmean(up))
    mean_down = float(np.nanmean(down))
    delta = mean_up - mean_down
    if normalize and mean_down != 0:
        delta = delta / abs(mean_down)

    return {"delta": float(delta), "mean_up": mean_up, "mean_down": mean_down}
