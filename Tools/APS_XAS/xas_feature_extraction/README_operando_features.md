# Operando/In-situ Decision Features

This folder contains standalone utilities for **time-series** (scan/sequence) analysis in operando or in-situ XAS.
Each module is independent and can be used without the main feature extractor.

## Expected Input
All functions accept a **pandas DataFrame** with one row per scan/measurement.
You provide the `feature_col` (e.g., `e0`, `white_line_intensity`) and any process metadata columns.

Typical columns you may have:
- `scan_index` or `time_s`
- `e0`, `white_line_intensity`, `edge_step`, etc.
- `temperature`, `pressure`, `flow_rate`
- `ramp_direction` (values like `up` / `down`)
- `snr` (signal-to-noise)

## Modules
- `xas_operando_temperature.py`
  - `temperature_correlation(df, feature_col, temp_col="temperature")`

- `xas_operando_process.py`
  - `process_correlation(df, feature_col, process_col)`

- `xas_operando_hysteresis.py`
  - `hysteresis_indicator(df, feature_col, condition_col="ramp_direction", up_value="up", down_value="down")`

- `xas_operando_stable_window.py`
  - `detect_stable_windows(df, feature_col, window_size=5, max_abs_slope=0.001, x_col=None)`

- `xas_operando_change_detection.py`
  - `detect_significant_changes(df, feature_col, zscore_thresh=2.0, min_delta=None, window=3)`

- `xas_operando_stop_criteria.py`
  - `stop_criteria(df, feature_col, window_size=5, max_range=0.002, snr_col=None, min_snr=None)`

## Minimal Examples
```python
import pandas as pd
from xas_operando_temperature import temperature_correlation
from xas_operando_change_detection import detect_significant_changes

# df has columns: time_s, e0, temperature, snr
corr = temperature_correlation(df, feature_col="e0", temp_col="temperature")
changes = detect_significant_changes(df, feature_col="e0", zscore_thresh=2.5)
```

```python
from xas_operando_stop_criteria import stop_criteria

stop = stop_criteria(df, feature_col="white_line_intensity", window_size=6, max_range=0.001,
                     snr_col="snr", min_snr=3.0)
if stop["stop"]:
    print("Stop recommended:", stop["reasons"])
```

## Notes
- These functions do **not** modify data; they return metrics/flags for your agent or UI layer.
- For correlation functions, if `scipy` is available, p-values are returned. Otherwise only `r` is reported.
- For change detection and stable windows, you can pass a time column via `x_col` to compute slopes in real units.
