# Standardized XAS Data Format Specification

**Version:** 3.0  
**Last Updated:** March 6, 2026

---

## Purpose

This document defines the **standardized xarray.Dataset format** that ALL XAS readers must output, regardless of input file format or beamline source. This ensures that downstream analysis tools (xas_analyzer, feature extraction, plotting, etc.) can process any XAS data without modification.

---

## Standard xarray.Dataset Structure

### Dimensions
```python
Dimensions: (point: N)
```
- **`point`**: Index along the energy scan (N data points)

### Coordinates (REQUIRED)
```python
Coordinates:
  * energy  (point) float64   # Photon energy [eV]
```

**Attributes for `energy` coordinate:**
- `units`: str = "eV" (always eV for calibrated data)
- `description`: str = "Photon energy"
- `calibrated`: bool = True if energy is calibrated, False if raw encoder/motor position

### Data Variables (REQUIRED)
```python
Data variables:
    i0        (point) float64   # Incident beam intensity
    i1        (point) float64   # Transmitted beam intensity  
    mu_trans  (point) float64   # Transmission absorption coefficient ln(I0/I1)
```

**Attributes for each data variable:**
- `description`: str - Human-readable description
- `units`: str - Physical units (if applicable)

### Data Variables (OPTIONAL)
```python
    i2           (point) float64   # Reference detector (foil)
    mu_ref       (point) float64   # Reference absorption ln(I1/I2)
    fluor_total  (point) float64   # Total fluorescence signal
    fluor_channel_N  (point) float64   # Individual fluorescence channels (N=1-7)
```

### Dataset Attributes (REQUIRED)
```python
Attributes:
    filename: str           # Original filename (without path)
    beamline: str          # Beamline identifier (e.g., "APS 12-BM-B", "SSRL 4-1")
    mode: str              # "transmission" or "fluorescence"
    n_points: int          # Number of data points
    energy_range: tuple    # (min_eV, max_eV)
    reader_version: str    # Version of reader module used
```

### Dataset Attributes (OPTIONAL)
```python
    date: str              # Collection date (if available)
    source: str            # Additional beamline/facility info
    format: str            # Original file format (e.g., "HDF5 (raw)", "ASCII", "XDI")
    calibrated: bool       # Energy calibration status
    energy_calibration: dict  # Calibration parameters if applicable
    header: list           # Original file header lines
    metadata: dict         # Additional beamline-specific metadata
```

---

## Data Variable Naming Conventions

### Detector Intensities
- `i0`: Incident beam intensity (before sample)
- `i1`: Transmitted beam intensity (after sample)
- `i2`: Reference detector (after reference foil)
- `i3`, `i4`, etc.: Additional detectors

### Absorption Coefficients
- `mu_trans`: Transmission absorption = ln(I0/I1)
- `mu_ref`: Reference absorption = ln(I1/I2)
- `mu_fluor`: Fluorescence absorption (normalized)

### Fluorescence Channels
- `fluor_total`: Sum of all fluorescence channels
- `fluor_channel_1` through `fluor_channel_7`: Individual detector elements

### Raw Data (if needed)
- `encoder`: Raw encoder/motor position
- Prefix with `raw_` if keeping uncalibrated data: `raw_energy`, `raw_encoder`

---

## Mode Detection Rules

**Transmission Mode:**
- Required: `i0`, `i1`, `mu_trans`
- Optional: `i2`, `mu_ref`

**Fluorescence Mode:**
- Required: `i0`, `fluor_total`
- Optional: `i1`, individual channels
- Calculated: `mu_fluor` = fluor_total / i0 (if needed)

**Auto-detection criteria:**
1. If `fluor_total` exists AND max(fluor_total) > 100 → "fluorescence"
2. Otherwise → "transmission"

---

## Energy Calibration Requirements

### Calibrated Energy (REQUIRED for analysis)
- Coordinate name: `energy`
- Units: `eV` (electron volts)
- Calibrated flag: `data['energy'].attrs['calibrated'] = True`

### Uncalibrated Energy (only if calibration unavailable)
- Coordinate name: `energy` (still use this name!)
- Units: `encoder_units` or `motor_units`
- Calibrated flag: `data['energy'].attrs['calibrated'] = False`
- Include warning in dataset attributes

**Calibration parameters (if applied):**
```python
data.attrs['energy_calibration'] = {
    'offset': float,  # eV
    'scale': float,   # eV per encoder unit
    'method': str,    # 'linear', 'reference_foil', 'known_edge', etc.
    'reference_edge': float  # eV (if using reference foil)
}
```

---

## Validation Checklist

Before returning a dataset, readers MUST ensure:

- [ ] `energy` coordinate exists with valid float64 values
- [ ] `i0`, `i1`, `mu_trans` data variables exist
- [ ] All required attributes present: `filename`, `beamline`, `mode`, `n_points`, `energy_range`
- [ ] Energy units specified: `data['energy'].attrs['units']`
- [ ] Mode correctly determined: "transmission" or "fluorescence"
- [ ] Energy range is realistic (typically 7000-8000 eV for Fe K-edge)
- [ ] No NaN or Inf values in required variables
- [ ] Array lengths match: len(energy) == len(i0) == len(i1) == len(mu_trans)

---

## Example: Minimal Valid Dataset

```python
import xarray as xr
import numpy as np

# Example: Fe K-edge transmission data
energy = np.linspace(7000, 7200, 300)
i0 = np.random.rand(300) * 1e6
i1 = np.random.rand(300) * 5e5
mu_trans = np.log(i0 / i1)

data = xr.Dataset(
    data_vars={
        'i0': ('point', i0, {'description': 'Incident beam intensity', 'units': 'counts'}),
        'i1': ('point', i1, {'description': 'Transmitted beam intensity', 'units': 'counts'}),
        'mu_trans': ('point', mu_trans, {'description': 'Transmission absorption coefficient ln(I0/I1)'})
    },
    coords={
        'energy': ('point', energy, {'units': 'eV', 'description': 'Photon energy', 'calibrated': True})
    },
    attrs={
        'filename': 'Fe_sample_001',
        'beamline': 'APS 12-BM-B',
        'mode': 'transmission',
        'n_points': 300,
        'energy_range': (7000.0, 7200.0),
        'reader_version': '3.0'
    }
)

# Validate
assert 'energy' in data.coords
assert all(var in data for var in ['i0', 'i1', 'mu_trans'])
assert all(attr in data.attrs for attr in ['filename', 'beamline', 'mode', 'n_points', 'energy_range'])
print("✓ Valid standardized XAS dataset")
```

---

## Example: Full Dataset with Optional Data

```python
data = xr.Dataset(
    data_vars={
        # REQUIRED
        'i0': ('point', i0_data, {'description': 'Incident beam intensity'}),
        'i1': ('point', i1_data, {'description': 'Transmitted beam intensity'}),
        'mu_trans': ('point', mu_trans_data, {'description': 'Transmission absorption ln(I0/I1)'}),
        
        # OPTIONAL - Reference foil
        'i2': ('point', i2_data, {'description': 'Reference detector intensity'}),
        'mu_ref': ('point', mu_ref_data, {'description': 'Reference absorption ln(I1/I2)'}),
        
        # OPTIONAL - Fluorescence
        'fluor_total': ('point', fluor_total_data, {'description': 'Total fluorescence'}),
        'fluor_channel_1': ('point', fluor1_data, {'description': 'Fluorescence channel 1'}),
        # ... more channels ...
    },
    coords={
        'energy': ('point', energy_eV, {
            'units': 'eV', 
            'description': 'Photon energy',
            'calibrated': True
        })
    },
    attrs={
        # REQUIRED
        'filename': 'FeCl2_sample',
        'beamline': 'APS 12-BM-B',
        'mode': 'fluorescence',
        'n_points': 333,
        'energy_range': (7012.0, 7212.0),
        'reader_version': '3.0',
        
        # OPTIONAL
        'date': '2026-03-06',
        'format': 'ASCII',
        'calibrated': True,
        'energy_calibration': {
            'offset': 7000.0,
            'scale': 1.0,
            'method': 'beamline_calibrated'
        }
    }
)
```

---

## Reader Implementation Guidelines

### 1. File Format Detection
Each reader should:
- Auto-detect file format (HDF5, ASCII, XDI, etc.)
- Handle multiple file extensions
- Provide format info in `attrs['format']`

### 2. Energy Calibration
- **Best practice:** Always calibrate to eV if possible
- If calibration unavailable: Set `calibrated=False` and include warning
- Store calibration method in attributes

### 3. Mode Detection
```python
# Auto-detect mode
if 'fluor_total' in data_vars and np.max(fluor_total) > 100:
    mode = 'fluorescence'
else:
    mode = 'transmission'
```

### 4. Data Quality Checks
```python
# Remove invalid data
mu_trans = np.log(i0 / i1)
mu_trans = np.nan_to_num(mu_trans, nan=0.0, posinf=0.0, neginf=0.0)

# Validate energy
assert np.all(np.diff(energy) > 0), "Energy must be monotonically increasing"
assert len(energy) > 10, "Need at least 10 data points"
```

### 5. Attribute Consistency
```python
# Always include reader version
attrs['reader_version'] = '3.0'

# Calculate energy range from actual data
attrs['energy_range'] = (float(energy.min()), float(energy.max()))
attrs['n_points'] = len(energy)
```

---

## Beamline-Specific Readers

### APS 12-BM-B (`aps_xas_reader.py`)
- Handles: HDF5 (raw), ASCII (processed)
- Fluorescence: Available in ASCII files (7 channels + total)
- Reference: i2 detector, mu_ref calculated
- Energy: Pre-calibrated in ASCII, needs calibration for HDF5

### Generic XAS Reader (`xas_reader.py`)
- Handles: XDI, generic ASCII (.dat, .txt, .csv)
- Supports: 2-4 column formats
- Auto-detects: Transmission vs fluorescence from headers
- Reference calibration: Optional (can be disabled)

### Future Readers
- SSRL beamlines
- ESRF beamlines
- ALS beamlines
- Custom lab sources

All must output the same standardized format!

---

## Integration with xas_analyzer

The analyzer expects this exact format:

```python
from xas_analyzer import XASAnalyzer

# Works with ANY reader output
data = load_xas_file(file_path)  # Returns standardized xr.Dataset

analyzer = XASAnalyzer()
result = analyzer.analyze(data)  # No format-specific code needed!

# Access data consistently
energy = data['energy'].values
mu = data['mu_trans'].values
filename = data.attrs['filename']
mode = data.attrs['mode']
```

---

## Version History

### v3.0 (March 6, 2026)
- Standardized format across all readers
- Added calibration metadata
- Defined validation checklist
- Added reader version tracking

### v2.0 (March 5, 2026)
- Switched to xarray.Dataset format
- Removed custom dataclasses

### v1.0 (March 2026)
- Initial format with custom dataclass

---

## Contact

For questions about format standardization, contact the ZZY Lab development team.

**Enforce this standard for ALL new readers!**
