# XAS Automated Data Processing & Validation Workflow

**Purpose:**  
This document defines a **production-grade, agent-safe, modular Python workflow** for automated processing, validation, and quality assessment of X-ray Absorption Spectroscopy (XAS) data (APS-compatible). It is intended to be read **before any code is written** by a coding agent.

The goal is **not merely data processing**, but **scientifically defensible automation** with explicit uncertainty, validation, and machine-readable quality signals.
# adapt availables moduels/functions from larch library: https://github.com/xraypy/xraylarch/blob/master/larch/math/normalization.py and this autoXAS: https://github.com/UlrikFriisJensen/autoXAS/blob/main/tutorials/autoXAS_exampleNotebook.ipynb
---

## 1. Design Principles (Mandatory)

- Every step must be implemented as an **independent, importable Python module**
- No step may fail silently
- Every automated decision must emit:
  - parameters
  - metrics
  - confidence score
  - flags
- The workflow must be:
  - scientifically conservative
  - agent-readable
  - safe for downstream automation

> Normalization, deglitching, and alignment are **hypotheses**, not facts.

---

## 2. Canonical Workflow Order (Do Not Change)

The following steps **must be executed in order**. Each step corresponds to one module.

```
raw data → reference (optional) → energy alignment → deglitching → normalization → validation → spectrum QC → plotting (optional)
```

---

## 3. Module Specifications

### 3.1 Raw Data Loader (`xas_reader`)

**Responsibilities**
- Load APS-compatible XAS formats (ASCII, XDI; HDF5 optional)
- Support transmission and fluorescence
- Standardize output representation

**Outputs**
- energy (eV)
- μ(E)
- metadata (beamline, scan ID, detector type, if available)

**Validation Checks**
- Energy monotonicity
- Missing or NaN values
- Zero-division in μ(E)

**Flags**
- `non_monotonic_energy`
- `missing_channels`
- `invalid_intensity`

---

### 3.2 Reference Loader (Optional) (`xas_reference_loader`)

**Responsibilities**
- Load reference spectrum (foil / standard)
- Validate energy range overlap
- Never block pipeline execution

**Outputs**
- reference_energy
- reference_mu
- reference_metadata

**Flags**
- `energy_range_mismatch`
- `reference_missing`

---

### 3.3 Energy Alignment (`energy_alignment`)

**Responsibilities**
- Estimate E₀ from derivative analysis
- Align sample E₀ to reference if provided
- Quantify confidence in alignment

**Outputs**
- ΔE applied
- method (`derivative` | `reference`)
- alignment confidence

**Failure Conditions**
- Multiple derivative maxima
- Ambiguous edge position

**Flags**
- `ambiguous_edge`
- `multiple_edges_detected`

---

### 3.4 Deglitching / Spike Removal (`deglitching`)

**Responsibilities**
- Remove detector spikes, Bragg peaks, glitches
- Use `larch.math.deglitch`
  - https://github.com/xraypy/xraylarch/blob/master/larch/math/deglitch.py
- Must be conservative and reversible

**Outputs**
- cleaned μ(E)
- glitch mask
- points removed count

**Flags**
- `excessive_deglitching`
- `localized_artifacts`

---

### 3.5 Normalization (`xas_normalization`)

**Responsibilities**
- Propose normalization parameters using:
  - quick signal diagnostics
  - physics-informed defaults
- Perform:
  - pre-edge subtraction
  - edge step normalization

**Stored Parameters**
- E₀
- pre-edge window
- post-edge window
- polynomial order

**Rules**
- No chemical assumptions unless explicitly provided
- Parameters must be explicitly recorded

---

### 3.6 Normalization Validation (`normalization_validator`)

**Physics-Based Validation Checks**
- Pre-edge flatness
- Edge step magnitude sanity
- Post-edge smoothness
- Sensitivity to window perturbation

**Outputs**
- validation metrics
- normalization confidence score

**Flags**
- `poor_preedge_fit`
- `unstable_edge_step`
- `postedge_noise`

---

### 3.7 Spectrum-Level Quality Assessment (`spectrum_quality_check`)

**Responsibilities**
- Detect globally poor-quality spectra

**Checks**
- Saturation
- Extremely low edge jump
- Excessive noise

**Classification**
- `usable`
- `usable_with_warning`
- `invalid`

**Outputs**
- quality classification
- quality confidence
- flags

---

### 3.8 Diagnostic Plotting (`quality_control`)

**Responsibilities**
- Generate diagnostic plots only (no publication figures)

**Plots**
- Raw μ(E)
- Deglitched μ(E)
- Normalized XANES
- Highlighted pre-edge / post-edge regions

**Rules**
- Optional execution
- No plotting inside physics modules

---

## 4. Output Contract (Agent-Readable)

The analyzer must emit a **single structured object**:

```json
{
  "energy_alignment": {
    "delta_e": 1.2,
    "confidence": 0.95
  },
  "deglitching": {
    "points_removed": 12,
    "flags": []
  },
  "normalization": {
    "parameters": {},
    "metrics": {},
    "confidence": 0.88,
    "flags": []
  },
  "spectrum_quality": {
    "classification": "usable",
    "confidence": 0.91,
    "flags": []
  }
}
```

This object is consumed by:
- agent router
- downstream XANES / EXAFS modules
- user-facing explanations

---

## 5. Explicit Non-Goals (Do NOT Implement for now, later will work on it)

- EXAFS fitting
- FEFF integration
- Chemical state inference
- Machine learning embeddings

This workflow is **strictly preprocessing + validation**.

---

## 6. Scientific Safety Rules

- Bad data must be flagged, not “fixed”
- Confidence must accompany every automated decision
- Downstream agents must be able to explain failures

> A valid output may be: “This spectrum is unreliable; here is why.”

---

## 7. Deliverables

- Modular Python files
- Clear function signatures
- No hardcoded paths
- Dependencies limited to:
  - numpy
  - scipy
  - matplotlib
  - xraylarch

Inline comments must explain **why**, not just **what**.

---

**End of Specification**

