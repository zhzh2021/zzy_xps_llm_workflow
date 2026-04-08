# APS XAS Reader Module Documentation

**Version:** 2.0 (xarray-based)  
**Beamline:** APS 12-BM-B  
**Author:** ZZY Lab  
**Date:** March 5, 2026

---

## Overview

The `aps_xas_reader` module provides a standardized interface for reading X-ray Absorption Spectroscopy (XAS) data from the Advanced Photon Source (APS) beamline 12-BM-B. The module automatically handles multiple file formats and returns data in a **standardized xarray.Dataset format**, ensuring consistent data structures regardless of the input file type.

### Key Features

✅ **Standardized Output**: All data returned as `xarray.Dataset` objects  
✅ **Multi-Format Support**: Reads both ASCII (processed) and HDF5 (raw) formats  
✅ **Format-Agnostic Code**: Your analysis code works with any beamline format  
✅ **Auto-Detection**: Automatically detects file format and loads appropriately  
✅ **Rich Metadata**: Preserves experimental metadata as dataset attributes  
✅ **Helper Functions**: Convenient extractors for transmission, fluorescence, and reference data  
✅ **Batch Loading**: Load multiple files from a directory with pattern matching  

---

## Installation & Requirements

### Dependencies

```python
import h5py          # HDF5 file reading
import numpy as np   # Numerical arrays
import xarray as xr  # Standardized scientific datasets
```

### Installation

```bash
pip install h5py numpy xarray
```

---

## Supported File Formats

### 1. ASCII Format (Recommended)

**File Extension**: None, `.dat`, or `.txt`  
**Data Type**: Processed XAS spectra with calibrated energy  
**Number of Points**: 293-333 (typical)  

**Column Structure** (16 columns):
```
# 1_E_eV 2_I0 3_I1 4_I2 5_I3 6_mu01 7_mu12 8_flatot 9_fla1 ... 16_enc
```

| Column | Variable | Description |
|--------|----------|-------------|
| 1 | E_eV | Photon energy in eV (calibrated) |
| 2 | I0 | Incident beam intensity |
| 3 | I1 | Transmitted beam intensity |
| 4 | I2 | Reference detector (foil) |
| 5 | I3 | Additional detector |
| 6 | mu01 | Transmission mu = ln(I0/I1) |
| 7 | mu12 | Reference mu = ln(I1/I2) |
| 8 | flatot | Total fluorescence (sum) |
| 9-15 | fla1-7 | Individual fluorescence channels |
| 16 | enc | Encoder position |

**Advantages**:
- Pre-calibrated energy in eV
- Pre-calculated absorption coefficients
- Fluorescence data included
- Ready for immediate XAS analysis

### 2. HDF5 Format (Raw Data)

**File Extension**: `.hdf`  
**Data Type**: Raw beamline data with encoder positions  
**Number of Points**: ~196 (typical)  

**HDF5 Datasets**:
- `INENC1.VAL.Mean`: Encoder position (raw counts)
- `FMC_IN.VAL1.Mean`: I0 detector current
- `FMC_IN.VAL2.Mean`: I1 detector current
- `FMC_IN.VAL3.Mean`: I2 detector current (reference)
- `FMC_IN.VAL4.Mean`: I3 detector current
- `COUNTER1.OUT.Value`: Counter output
- `PCAP.SAMPLES.Value`: Sample count

**Limitations**:
- Requires energy calibration (encoder → eV conversion)
- No pre-calculated absorption coefficients
- No fluorescence data
- Lower resolution than ASCII files

---

## Standardized Dataset Structure

### xarray.Dataset Format

All files are converted to this standardized structure:

```python
<xarray.Dataset>
Dimensions:      (point: N)
Coordinates:
  * energy       (point) float64   # Photon energy [eV]
Data variables:
    i0           (point) float64   # Incident intensity
    i1           (point) float64   # Transmitted intensity
    mu_trans     (point) float64   # Transmission mu = ln(I0/I1)
    i2           (point) float64   # Reference detector [optional]
    mu_ref       (point) float64   # Reference mu = ln(I1/I2) [optional]
    fluor_total  (point) float64   # Total fluorescence [ASCII only]
    fluor_channel_1 ... fluor_channel_7  # Individual channels [ASCII only]
Attributes:
    filename: str                  # Original filename
    beamline: 'APS 12-BM-B'
    mode: 'transmission' | 'fluorescence'
    n_points: int                  # Number of data points
    energy_range: tuple            # (min_eV, max_eV)
    date: str                      # Collection date [if available]
    source: str                    # Beamline info
    format: str                    # 'HDF5 (raw)' for HDF files
    calibrated: bool               # Energy calibration status [HDF only]
```

### Coordinate System

- **Dimension**: `point` - index along the energy scan
- **Coordinate**: `energy` - photon energy values with units attribute

### Data Variables

**Required** (present in all datasets):
- `i0`, `i1`, `mu_trans`

**Optional** (format-dependent):
- `i2`, `mu_ref` - present if reference foil data available
- `fluor_*` - present only in ASCII files with fluorescence data

### Attributes

Metadata stored as dataset attributes (`ds.attrs`):
- `filename`: Original file name
- `beamline`: Source beamline identifier
- `mode`: Measurement mode
- `n_points`: Number of data points
- `energy_range`: Energy scan range tuple
- Additional format-specific metadata

---

## Basic Usage

### Loading a Single File

```python
from aps_xas_reader import load_aps_xas

# Load ASCII file (automatically detected)
data = load_aps_xas('FeCl2-Malic_acid_pH2.2')

# Load HDF5 file
data = load_aps_xas('sample.hdf')

# Load HDF5 with energy calibration
data = load_aps_xas('sample.hdf', energy_calibration=(7000.0, 1e-7))
# calibration: energy_eV = 7000.0 + 1e-7 * encoder_position
```

### Accessing Data

```python
# Extract arrays
energy = data['energy'].values      # Energy in eV
mu = data['mu_trans'].values        # Absorption coefficient
i0 = data['i0'].values              # Incident intensity

# Access metadata
filename = data.attrs['filename']
mode = data.attrs['mode']
n_points = data.attrs['n_points']
energy_range = data.attrs['energy_range']

# Check optional data
if 'mu_ref' in data:
    mu_ref = data['mu_ref'].values  # Reference absorption

if 'fluor_total' in data:
    fluor = data['fluor_total'].values  # Fluorescence signal
```

### Using Helper Functions

```python
from aps_xas_reader import (
    get_transmission_mu,
    get_fluorescence_mu,
    get_reference_mu
)

# Extract transmission data
energy, mu_trans = get_transmission_mu(data)

# Extract fluorescence data (normalized by I0)
energy, mu_fluor = get_fluorescence_mu(data, normalize=True)

# Extract reference foil data
energy, mu_ref = get_reference_mu(data)
```

### Batch Loading Multiple Files

```python
from aps_xas_reader import load_aps_dataset

# Load all FeCl2 samples
datasets = load_aps_dataset(
    r"N:\data\xas_data",
    pattern="FeCl2*",
    prefer_ascii=True
)

print(f"Loaded {len(datasets)} files")

# Process all datasets
for ds in datasets:
    energy = ds['energy'].values
    mu = ds['mu_trans'].values
    sample_name = ds.attrs['filename']
    # ... your analysis code ...
```

---

## Advanced Usage

### Working with xarray Features

```python
# Select energy range
subset = data.sel(point=slice(100, 200))

# Select by energy value (interpolation)
data_interp = data.interp(energy=7120.0)

# Plotting with xarray
import matplotlib.pyplot as plt

data['mu_trans'].plot(x='energy')
plt.xlabel('Energy (eV)')
plt.ylabel('μ(E)')
plt.title(data.attrs['filename'])
plt.show()

# Statistical operations
mu_mean = data['mu_trans'].mean()
mu_std = data['mu_trans'].std()

# Combine multiple datasets
import xarray as xr
combined = xr.concat(datasets, dim='sample')
```

### Energy Calibration for HDF Files

If you have HDF5 files and know the energy calibration:

```python
# Calibration from known edge position
# For example, Fe K-edge at 7112 eV corresponds to encoder 71120000
offset = 7112.0
scale = 1e-4  # Example: 0.0001 eV per encoder count

data = load_aps_xas(
    'sample.hdf',
    energy_calibration=(offset, scale)
)

# Verify calibration
print(f"Energy range: {data.attrs['energy_range']}")
```

### Filtering and Preprocessing

```python
import numpy as np

# Remove pre-edge offset
pre_edge_region = (data['energy'] < 7100)
pre_edge_mu = data['mu_trans'].where(pre_edge_region).mean()
data['mu_trans_normalized'] = data['mu_trans'] - pre_edge_mu

# Moving average smoothing
from scipy.ndimage import uniform_filter1d

mu_smooth = uniform_filter1d(data['mu_trans'].values, size=5)
data['mu_trans_smooth'] = ('point', mu_smooth)
```

---

## Module API Reference

### Core Functions

#### `load_aps_xas(file_path, prefer_ascii=True, energy_calibration=None)`

Main loading function with automatic format detection.

**Parameters**:
- `file_path` (str | Path): Path to XAS data file
- `prefer_ascii` (bool): Prefer ASCII over HDF if both exist
- `energy_calibration` (tuple): Optional `(offset, scale)` for HDF files

**Returns**: `xarray.Dataset`

**Example**:
```python
data = load_aps_xas('sample.hdf', energy_calibration=(7000, 1e-7))
```

---

#### `read_aps_ascii(file_path)`

Read ASCII format XAS data.

**Parameters**:
- `file_path` (str | Path): Path to ASCII file

**Returns**: `xarray.Dataset`

**Example**:
```python
data = read_aps_ascii('FeCl2_sample')
```

---

#### `read_aps_hdf(file_path, energy_calibration=None)`

Read HDF5 format XAS data.

**Parameters**:
- `file_path` (str | Path): Path to HDF5 file
- `energy_calibration` (tuple): Optional `(offset, scale)`

**Returns**: `xarray.Dataset`

**Example**:
```python
data = read_aps_hdf('sample.hdf')
```

---

### Helper Functions

#### `get_transmission_mu(data)`

Extract transmission mode data.

**Parameters**:
- `data` (xr.Dataset): XAS dataset

**Returns**: `(energy, mu)` tuple of numpy arrays

**Example**:
```python
energy, mu = get_transmission_mu(data)
```

---

#### `get_fluorescence_mu(data, normalize=True)`

Extract fluorescence mode data.

**Parameters**:
- `data` (xr.Dataset): XAS dataset
- `normalize` (bool): Divide by I0 if True

**Returns**: `(energy, mu)` tuple of numpy arrays

**Raises**: `ValueError` if no fluorescence data

**Example**:
```python
energy, mu_fluor = get_fluorescence_mu(data, normalize=True)
```

---

#### `get_reference_mu(data)`

Extract reference foil data.

**Parameters**:
- `data` (xr.Dataset): XAS dataset

**Returns**: `(energy, mu_ref)` tuple of numpy arrays

**Raises**: `ValueError` if no reference data

**Example**:
```python
energy, mu_ref = get_reference_mu(data)
```

---

### Batch Loading

#### `load_aps_dataset(data_dir, pattern="*", prefer_ascii=True)`

Load multiple files from a directory.

**Parameters**:
- `data_dir` (str | Path): Directory containing XAS files
- `pattern` (str): Glob pattern for file matching
- `prefer_ascii` (bool): Prefer ASCII over HDF

**Returns**: `list[xr.Dataset]`

**Example**:
```python
datasets = load_aps_dataset(
    r"N:\data\xas",
    pattern="FeCl2*",
    prefer_ascii=True
)
```

---

## Workflow Examples

### Example 1: Simple XANES Analysis

```python
from aps_xas_reader import load_aps_xas
import matplotlib.pyplot as plt

# Load data
data = load_aps_xas('FeCl2_sample')

# Extract transmission data
energy = data['energy'].values
mu = data['mu_trans'].values

# Plot XANES
plt.figure(figsize=(8, 6))
plt.plot(energy, mu, 'b-', linewidth=2)
plt.xlabel('Energy (eV)', fontsize=12)
plt.ylabel('μ(E)', fontsize=12)
plt.title(f"XANES: {data.attrs['filename']}", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('xanes_plot.png', dpi=300)
plt.show()
```

### Example 2: Comparing Multiple Samples

```python
from aps_xas_reader import load_aps_dataset
import matplotlib.pyplot as plt

# Load all samples with FeCl2
datasets = load_aps_dataset('.', pattern='FeCl2*')

# Plot all on same axes
plt.figure(figsize=(10, 7))
for ds in datasets:
    energy = ds['energy'].values
    mu = ds['mu_trans'].values
    label = ds.attrs['filename']
    plt.plot(energy, mu, label=label, alpha=0.7)

plt.xlabel('Energy (eV)')
plt.ylabel('μ(E)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

### Example 3: Fluorescence vs Transmission

```python
from aps_xas_reader import load_aps_xas, get_transmission_mu, get_fluorescence_mu
import matplotlib.pyplot as plt

data = load_aps_xas('fluorescence_sample')

# Get both modes
energy_trans, mu_trans = get_transmission_mu(data)
energy_fluor, mu_fluor = get_fluorescence_mu(data, normalize=True)

# Compare
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(energy_trans, mu_trans, 'b-')
ax1.set_ylabel('Transmission μ(E)')
ax1.set_title('Transmission Mode')
ax1.grid(True, alpha=0.3)

ax2.plot(energy_fluor, mu_fluor, 'r-')
ax2.set_ylabel('Fluorescence μ(E)')
ax2.set_xlabel('Energy (eV)')
ax2.set_title('Fluorescence Mode (I_fluor/I0)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Example 4: Export to CSV with Metadata

```python
from aps_xas_reader import load_aps_xas
import pandas as pd

data = load_aps_xas('sample')

# Create DataFrame
df = pd.DataFrame({
    'energy_eV': data['energy'].values,
    'mu_transmission': data['mu_trans'].values,
    'i0': data['i0'].values,
    'i1': data['i1'].values
})

# Add optional columns
if 'mu_ref' in data:
    df['mu_reference'] = data['mu_ref'].values

if 'fluor_total' in data:
    df['fluorescence_total'] = data['fluor_total'].values

# Save with metadata header
with open('output.csv', 'w') as f:
    f.write(f"# Sample: {data.attrs['filename']}\n")
    f.write(f"# Beamline: {data.attrs['beamline']}\n")
    f.write(f"# Mode: {data.attrs['mode']}\n")
    f.write(f"# Points: {data.attrs['n_points']}\n")
    f.write(f"# Energy Range: {data.attrs['energy_range']}\n")
    f.write("#\n")

df.to_csv('output.csv', mode='a', index=False)
```

---

## Benefits of xarray Format

### 1. **Self-Describing Data**
```python
# No need to remember array order or units
data['energy']  # Clear what this is
data['energy'].attrs['units']  # Units are attached
```

### 2. **Labeled Indexing**
```python
# Select by coordinate value, not array index
subset = data.sel(energy=slice(7100, 7150))
```

### 3. **Automatic Alignment**
```python
# Arithmetic on datasets with different grids
result = data1 + data2  # Automatically aligns by coordinates
```

### 4. **Built-in Plotting**
```python
# One-line plots with labels
data['mu_trans'].plot(x='energy')
```

### 5. **NetCDF Export**
```python
# Save complete dataset with metadata
data.to_netcdf('sample.nc')

# Load later
data_loaded = xr.open_dataset('sample.nc')
```

### 6. **Pandas Integration**
```python
# Convert to DataFrame when needed
df = data.to_dataframe()
```

---

## Best Practices

### ✅ DO:
- Use `load_aps_xas()` for automatic format detection
- Prefer ASCII files for XAS analysis (pre-calibrated)
- Check for optional data variables before accessing
- Preserve dataset attributes when processing
- Use helper functions for common data extraction

### ❌ DON'T:
- Assume all datasets have fluorescence data
- Mix uncalibrated HDF data with calibrated ASCII data
- Modify original data in-place (create new variables instead)
- Forget to check energy calibration for HDF files

---

## Troubleshooting

### Issue: "No fluorescence data available"

**Cause**: File is HDF5 format (raw data only)  
**Solution**: Use ASCII files or access transmission data instead

```python
# Instead of:
energy, mu = get_fluorescence_mu(data)  # Error!

# Use:
energy, mu = get_transmission_mu(data)  # Works for all formats
```

### Issue: Energy values look wrong (very large/small)

**Cause**: HDF5 file loaded without energy calibration  
**Solution**: Provide calibration parameters or use ASCII file

```python
# Check energy units
print(data['energy'].attrs['units'])

# If 'encoder_units', need calibration
data = load_aps_xas('file.hdf', energy_calibration=(7000, 1e-7))
```

### Issue: Different number of points between files

**Cause**: ASCII files have more points than HDF5 (normal)  
**Solution**: Interpolate to common grid if comparing

```python
import numpy as np
import xarray as xr

# Create common energy grid
e_min = max(ds['energy'].min().values for ds in datasets)
e_max = min(ds['energy'].max().values for ds in datasets)
common_energy = np.linspace(e_min, e_max, 300)

# Interpolate all datasets
aligned = [ds.interp(energy=common_energy) for ds in datasets]
```

---

## File Naming Conventions

Based on your experimental setup:

**Pattern**: `{IronSource}-{Ligand}_{Concentration}-pH{pH}`

Examples:
- `FeCl2-Malic_acid_(0.5-0.5)-pH2.2`
- `FeSO4-Tartaric_acid_(0.5-0.5)-pH5.1`

**Metadata Extraction**: Use `map_filename.py` to parse conditions from filenames.

---

## Version History

### Version 2.0 (March 5, 2026)
- ✨ **Major**: Switched to xarray.Dataset as standardized output format
- 🔧 Removed APSXASData dataclass
- ✨ Added coordinate and attribute metadata
- 🔧 Updated all helper functions for xarray compatibility
- ✨ Improved format consistency between ASCII and HDF5
- 📝 Comprehensive documentation

### Version 1.0 (March 2026)
- Initial release with APSXASData dataclass
- Support for ASCII and HDF5 formats
- Basic helper functions

---

## Contact & Support

**Lab**: ZZY Autonomous Lab  
**Beamline**: APS 12-BM-B  
**Data Location**: `N:\zhenzhen\C-Steel\Data\XAS data\`

For issues or questions about this module, refer to the test suite in `test_xarray_reader.py` for working examples.

---

## License

Internal use for ZZY Lab research projects.

---

**Last Updated**: March 5, 2026  
**Module Version**: 2.0  
**Python Version**: 3.11+
