# XAS Reader Compatibility Guide

**Version:** 3.0  
**Date:** March 6, 2026

---

## Quick Reference: All Readers Output Same Format

No matter which beamline or file format you use, **all readers output identical `xarray.Dataset` structure**. Your analysis code works everywhere!

---

## Unified API

### Loading Data (Same Interface)

```python
# APS 12-BM-B Reader
from aps_xas_reader import load_aps_xas
data_aps = load_aps_xas('FeCl2_sample')  # Returns xr.Dataset

# Generic Reader
from xas_reader import load_xas_file
data_generic = load_xas_file('sample.dat', beamline='SSRL 4-1')  # Returns xr.Dataset

# Both return IDENTICAL structure!
```

### Accessing Data (Same Code)

```python
# Works with ANY reader output
energy = data['energy'].values
mu = data['mu_trans'].values
i0 = data['i0'].values
i1 = data['i1'].values

# Check metadata
filename = data.attrs['filename']
beamline = data.attrs['beamline']
mode = data.attrs['mode']
n_points = data.attrs['n_points']
```

### Helper Functions (Same API)

```python
# Works with data from ANY beamline
from aps_xas_reader import get_transmission_mu, get_fluorescence_mu, get_reference_mu

# Or
from xas_reader import get_transmission_mu, get_fluorescence_mu, get_reference_mu

# Same functions, same signatures!
energy, mu = get_transmission_mu(data)
```

---

## Reader Comparison

| Feature | aps_xas_reader.py | xas_reader.py |
|---------|-------------------|---------------|
| **Format** | xarray.Dataset | xarray.Dataset |
| **Beamline** | APS 12-BM-B | Generic (any) |
| **Input Formats** | HDF5, ASCII | XDI, DAT, TXT, CSV |
| **Energy Coordinate** | `energy` (eV) | `energy` (eV) |
| **Required Variables** | i0, i1, mu_trans | i0, i1, mu_trans |
| **Optional Variables** | i2, mu_ref, fluor_* | i2, mu_ref, fluor_total |
| **Metadata** | Same attributes | Same attributes |
| **Version** | 3.0 | 3.0 |

---

## Example: Switching Beamlines (No Code Changes)

```python
import matplotlib.pyplot as plt

# Load data from DIFFERENT beamlines
from aps_xas_reader import load_aps_xas
from xas_reader import load_xas_file

aps_data = load_aps_xas('APS_sample')
ssrl_data = load_xas_file('SSRL_sample.dat', beamline='SSRL 4-1')

# SAME plotting code works for BOTH!
for data, label in [(aps_data, 'APS'), (ssrl_data, 'SSRL')]:
    plt.plot(data['energy'], data['mu_trans'], label=label)

plt.legend()
plt.xlabel('Energy (eV)')
plt.ylabel('μ(E)')
plt.show()
```

---

## Example: Analyzer Integration

```python
from xas_analyzer import XASAnalyzer

# Analyzer doesn't care about beamline!
analyzer = XASAnalyzer()

# Works with APS data
aps_result = analyzer.analyze(aps_data)

# Works with SSRL data
ssrl_result = analyzer.analyze(ssrl_data)

# Works with ANY properly formatted data
```

---

## Format Validation

Both readers include validation:

```python
# After loading, verify format
assert 'energy' in data.coords
assert all(var in data for var in ['i0', 'i1', 'mu_trans'])
assert 'reader_version' in data.attrs
print(f"✓ Valid dataset from {data.attrs['reader_name']}")
```

---

## Migration Guide

### Old Code (Before v3.0)
```python
# Old tuple return format
energy, mu = load_xas_file('sample.dat')

# Plotting
plt.plot(energy, mu)
```

### New Code (v3.0+)
```python
# New xarray.Dataset format
data = load_xas_file('sample.dat')

# Plotting - extract arrays
plt.plot(data['energy'], data['mu_trans'])

# Or use xarray plotting
data['mu_trans'].plot(x='energy')
```

### Why Change?

**Benefits of xarray format:**
1. **Self-describing** - Data carries metadata
2. **Consistent** - Same structure everywhere
3. **Flexible** - Easy to add new variables
4. **Powerful** - Built-in operations (interpolation, slicing, etc.)
5. **Standard** - Used by scientific Python community

---

## Reader-Specific Features

### APS Reader (`aps_xas_reader.py`)

**Handles:**
- HDF5 files with raw encoder data
- ASCII files with pre-processed data
- 7 fluorescence channels + total

**Energy Calibration:**
```python
# For HDF5 files only
data = load_aps_xas('sample.hdf', 
                    energy_calibration=(7000.0, 1e-7))
```

**Batch Loading:**
```python
from aps_xas_reader import load_aps_dataset

datasets = load_aps_dataset(
    r"N:\data\xas",
    pattern="FeCl2*",
    prefer_ascii=True
)
```

### Generic Reader (`xas_reader.py`)

**Handles:**
- XDI format (.xdi)
- Generic ASCII (2, 3, or 4 columns)
- Auto-detects transmission vs fluorescence

**Column Formats:**
- 2 columns: `energy, mu`
- 3 columns: `energy, i0, it` (transmission) OR `energy, i0, iff` (fluorescence)
- 4 columns: `energy, i0, it, ir` (with reference)

**Batch Loading:**
```python
from xas_reader import load_xas_batch

datasets = load_xas_batch(
    '/path/to/data',
    pattern="*.dat",
    beamline="SSRL 4-1"
)
```

---

## Adding New Readers

To add a reader for a new beamline:

1. **Import requirements:**
```python
import numpy as np
import xarray as xr
from pathlib import Path
```

2. **Follow the specification:**
- See `STANDARDIZED_FORMAT_SPEC.md`
- Return `xr.Dataset` with required variables
- Include metadata attributes

3. **Use helper template:**
```python
def load_new_beamline(file_path):
    # Read your format
    energy, i0, i1 = read_your_format(file_path)
    
    # Calculate mu
    mu = np.log(i0 / i1)
    
    # Create standardized dataset
    data = xr.Dataset(
        data_vars={
            'i0': ('point', i0, {'description': 'Incident intensity'}),
            'i1': ('point', i1, {'description': 'Transmitted intensity'}),
            'mu_trans': ('point', mu, {'description': 'Absorption coefficient'})
        },
        coords={'energy': ('point', energy, {'units': 'eV', 'calibrated': True})},
        attrs={
            'filename': file_path.name,
            'beamline': 'New Beamline',
            'mode': 'transmission',
            'n_points': len(energy),
            'energy_range': (energy.min(), energy.max()),
            'reader_version': '3.0'
        }
    )
    
    return data
```

4. **Test compatibility:**
```python
data = load_new_beamline('test_file')
energy, mu = get_transmission_mu(data)  # Should work!
```

---

## Summary

✅ **All readers output `xarray.Dataset`**  
✅ **Same coordinate names** (`energy`)  
✅ **Same data variable names** (`i0`, `i1`, `mu_trans`, etc.)  
✅ **Same attributes** (`filename`, `beamline`, `mode`, etc.)  
✅ **Same helper functions** (`get_transmission_mu`, etc.)  
✅ **Works with any analyzer** (xas_analyzer, feature extractors, etc.)

**Result:** Write your analysis code once, use everywhere! 🎉

---

## Contact

For questions about adding new readers or format compatibility:
- See: `STANDARDIZED_FORMAT_SPEC.md`
- Check: Reader test files (`test_*.py`)
- Contact: ZZY Lab development team

**Always use standardized format v3.0+**
