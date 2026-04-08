# XAS Linear Combination Fitting (LCF) Module

This module performs Linear Combination Fitting (LCF) on XAS spectra to estimate reference fractions (e.g., Fe²⁺/Fe³⁺) with constraints, uncertainty estimates, and publication‑ready outputs.

File: `C:\GitRepos\zz_llm\zzy_llm\Tools\APS_XAS\xas_fitting\xas_LCF_fitting.py`

---

## What It Does

1. **Pre-edge normalization** for sample and references.
2. **Energy alignment** of references via interpolation onto the sample energy grid.
3. **Constrained fitting**:
   - Fractions are **non‑negative**.
   - Fractions **sum to 1**.
4. Optional **scale + offset** to mitigate normalization differences.
5. **Weighted fitting** to emphasize pre‑edge, XANES, or EXAFS regions.
6. **Bootstrap uncertainty** (residual resampling) for fraction confidence intervals.
7. Saves **full outputs** (JSON, CSV, NPZ, plots).

---

## Quick Start (Single Fit)

```python
from pathlib import Path
import pandas as pd
from xas_fitting.xas_LCF_fitting import perform_xas_lcf_fitting

def load_csv(p):
    df = pd.read_csv(p)
    energy = df['energy'].values
    mu = df['mu_normalized'].values if 'mu_normalized' in df.columns else df['mu_cleaned'].values
    return energy, mu

sample_energy, sample_mu = load_csv(r"C:\path\to\sample.csv")
ref2_energy, ref2_mu = load_csv(r"C:\path\to\ref_fe2.csv")
ref3_energy, ref3_mu = load_csv(r"C:\path\to\ref_fe3.csv")

res = perform_xas_lcf_fitting(
    sample_mu=sample_mu,
    energy=sample_energy,
    ref_mu_fe2=ref2_mu,
    ref_mu_fe3=ref3_mu,
    ref_energy_fe2=ref2_energy,
    ref_energy_fe3=ref3_energy,
    sample_name="sample_001",
    fit_region="xanes",
    weight_mode="pre_edge",
    weight_factor=3.0,
    save_outputs=True
)
```

## Reference Library (Recommended)

Place curated references in:
- Tool-level: `C:\GitRepos\zz_llm\zzy_llm\Tools\APS_XAS\xas_config\standards_library`
- Project override (preferred if exists): `C:\GitRepos\zz_llm\zzy_llm\project_root\standards_library`

Configure the library in:
`C:\GitRepos\zz_llm\zzy_llm\Tools\APS_XAS\xas_config\reference_library.yaml`

Example usage:
```python
from xas_fitting.xas_LCF_fitting import perform_batch_lcf_analysis_from_library

batch_results = perform_batch_lcf_analysis_from_library(
    samples_data=my_samples_dict,
    ref_labels=("Fe2+", "Fe3+")
)
```

---

## Multi‑Reference Fit (N‑References)

```python
from xas_fitting.xas_LCF_fitting import perform_xas_lcf_fitting_multi

refs = [
    {"label": "Fe2+", "mu": ref2_mu, "energy": ref2_energy, "shift": 0.1},
    {"label": "Fe3+", "mu": ref3_mu, "energy": ref3_energy, "shift": 0.0},
    {"label": "FeO",  "mu": ref_mu_feo, "energy": ref_energy_feo}
]

res = perform_xas_lcf_fitting_multi(
    sample_mu=sample_mu,
    energy=sample_energy,
    ref_spectra=refs,
    fit_region="xanes",
    weight_mode="pre_edge",
    weight_factor=3.0,
    save_outputs=True
)
```

---

## Key Parameters

### Fit Range Controls
- `fit_region`: `"pre_edge" | "xanes" | "exafs" | "custom"`
- `fit_range`: explicit `(emin, emax)` when using `"custom"`
- `adaptive_window`: use percentile windows (data‑driven)
- `window_percentiles`: default `(5, 95)`

### Weighting
- `weight_mode`: `"uniform"`, `"pre_edge"`, `"xanes"`, `"exafs"`
- `weight_window`: override weighting window explicitly
- `weight_factor`: multiplier inside weighting window

### Scale + Offset
Helps compensate for small normalization differences.
- `allow_scale_offset`: default `True`
- `scale_bounds`: default `(0.9, 1.1)`
- `offset_bounds`: default `(-0.05, 0.05)`

### Bootstrap Uncertainty
- `bootstrap_samples`: default `200`
- `bootstrap_seed`: default `42`

---

## Outputs (Per Sample)

Saved automatically to:
- Data:  
  `C:\GitRepos\zz_llm\zzy_llm\project_root\xas_results\05_LCF_fitting\fitting_data`
- Plots:  
  `C:\GitRepos\zz_llm\zzy_llm\project_root\xas_results\05_LCF_fitting\fitting_plots`

Files:
- `*_lcf_results.json` — all config, normalization, fit range, diagnostics
- `*_lcf_fit_arrays.npz` — energy, sample, model, residuals
- `*_lcf_fit_table.csv` — tabular curves
- `*_lcf_fit.png` — overlay plot
- `*_lcf_residuals.png` — residual plot

---

## Batch Mode Outputs

Batch summary and index:
- `lcf_batch_summary.csv`
- `lcf_outputs_index.json`

These are written to:
`C:\GitRepos\zz_llm\zzy_llm\project_root\xas_results\05_LCF_fitting\fitting_data`

---

## Interpretation Tips

- **R‑factor** < 0.01 is generally good for XANES LCF.
- Check **residual plot** for systematic structure.
- If fractions look unstable:
  - Increase bootstrap samples
  - Tighten `fit_range`
  - Increase weighting in pre‑edge or white‑line region

---

## Notes

- This module assumes all spectra are already normalized.
- Reference shifts (`shift` in eV) are applied before interpolation.
- If using **non‑Fe edges**, set `e0` and ranges accordingly.
