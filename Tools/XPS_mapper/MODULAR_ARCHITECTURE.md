# XPS Map Processor - Modular Architecture Guide

## Overview
The XPS map processor has been refactored into a modular architecture for better maintainability, reusability, and testing. This guide documents the module organization and usage patterns.

## Module Structure

### Core Processing Modules

#### 1. **chemometrics_utils.py**
**Purpose**: Core chemometrics algorithms for hyperspectral data analysis

**Functions**:
- `compute_pre_image(cube)` - Pattern Recognition Entropy calculation
- `normalize_l1(X)` - L1 (sum) normalization for spectra
- `mask_low_counts(cube, threshold)` - Filter low-signal pixels
- `charge_align_cube(cube, energy, ref_energy)` - Correct charging effects
- `run_mcr_on_cube(cube, n_components)` - MCR-ALS/NMF decomposition
- `compute_spectral_variability(cube)` - Variability metrics

**When to use**:
- Preprocessing hyperspectral data
- Correcting experimental artifacts (charging)
- Decomposing mixed spectra into pure components
- Assessing data quality before analysis

**Dependencies**: numpy, sklearn.decomposition.NMF, pymcr (optional)

**Example**:
```python
from chemometrics_utils import compute_pre_image, run_mcr_on_cube

# Compute spectral complexity
pre_image = compute_pre_image(hyperspectral_cube)

# Decompose into 3 pure components
mcr_results = run_mcr_on_cube(hyperspectral_cube, n_components=3)
component_spectra = mcr_results['component_spectra']
concentration_maps = mcr_results['conc_maps']
```

---

#### 2. **chemometrics_plots.py**
**Purpose**: Visualization functions for chemometrics analysis

**Functions**:
- `plot_spectra_waterfall(cube, energy)` - Overlay/waterfall plot for initial inspection
- `plot_pre_image(pre_image)` - PRE map with histogram and statistics
- `plot_mcr_components(energy, spectra, maps)` - MCR/NMF component visualization

**When to use**:
- Initial data quality assessment (waterfall plot)
- Visualizing spatial heterogeneity (PRE map)
- Interpreting decomposition results (MCR plots)

**Dependencies**: matplotlib, numpy

**Example**:
```python
from chemometrics_plots import plot_spectra_waterfall, plot_pre_image

# Check if data has sufficient variability
variability = plot_spectra_waterfall(
    cube, energy, output_dir,
    n_spectra=20,
    plot_mode="overlay"
)
print(f"Mean variability: {variability['mean_std']}")

# Visualize spectral complexity map
plot_pre_image(pre_image, output_dir)
```

---

#### 3. **case1_2d_processing.py**
**Purpose**: Single-energy 2D map processing (intensity maps without energy dimension)

**Functions**:
- `compute_net_and_ratio(on_map, off_map)` - Background subtraction
- `denoise_map(data, sigma)` - Gaussian smoothing
- `threshold_segment(data, method)` - Binary segmentation
- `morph_cleanup(mask, op, size)` - Morphological operations
- `roi_stats(data, mask)` - ROI statistics
- `process_2d_map(parsed)` - Complete 2D workflow

**When to use**:
- Single-energy mapping (survey scans, specific peak intensity)
- ROI identification and segmentation
- Quick spatial distribution analysis

---

#### 4. **case2_hyperspectral_processing.py**
**Purpose**: Hyperspectral map processing (3D cubes with energy axis)

**Functions**:
- `baseline_als(y, lam, p)` - Asymmetric least squares baseline
- `fit_average_spectrum(energy, intensity)` - Peak fitting on average
- `pca_cluster_analysis(cube)` - PCA + clustering
- `process_hyperspectral(map_data)` - Complete hyperspectral workflow

**When to use**:
- Full spectral information at each pixel
- Phase identification via clustering
- Quantitative analysis (peak areas, shifts)

---

### Visualization Modules

#### 5. **map_plots_basic.py**
**Purpose**: Basic plotting utilities for 2D and hyperspectral maps

**Functions**:
- `plot_2d_overview(parsed, denoised, mask)` - 2D map summary
- `plot_average_spectrum(energy, intensity)` - Average spectrum plot
- `plot_area_maps(area_maps)` - Peak area maps
- `plot_shift_mse_maps(shift, mse)` - Fitting quality maps

---

#### 6. **component_plots.py**
**Purpose**: PCA and NMF component visualization

**Functions**:
- `plot_pca_components(energy, components, scores)` - PCA loadings + score maps
- `plot_nmf_components(energy, components, abundances)` - NMF components + abundance maps
- `plot_scree(explained_variance)` - Scree plot for component selection

---

#### 7. **cluster_plots.py**
**Purpose**: Clustering analysis visualization

**Functions**:
- `plot_cluster_analysis(map_data, cluster_results)` - Comprehensive cluster summary
- `plot_cluster_map(labels)` - Spatial cluster distribution
- `plot_cluster_spectra(energy, cluster_info)` - Representative spectra per cluster
- `plot_dendrogram(linkage_matrix)` - Hierarchical clustering tree

---

## Workflow Patterns

### Pattern 1: Quick Data Inspection (Before Deep Analysis)
```python
from chemometrics_plots import plot_spectra_waterfall
from chemometrics_utils import compute_spectral_variability

# 1. Visual check for variability
variability = plot_spectra_waterfall(cube, energy, output_dir)

# 2. Quantitative variability metric
metrics = compute_spectral_variability(cube)

# 3. Decision logic
if metrics['mean_std'] < 10:
    print("LOW variability - skip chemometrics")
else:
    print("SUFFICIENT variability - proceed with PCA/clustering")
```

### Pattern 2: Full Chemometrics Pipeline
```python
from chemometrics_utils import (
    compute_pre_image, mask_low_counts,
    charge_align_cube, run_mcr_on_cube
)
from chemometrics_plots import plot_pre_image, plot_mcr_components

# 1. Compute PRE
pre_image = compute_pre_image(cube)
plot_pre_image(pre_image, output_dir)

# 2. Filter low-signal pixels
mask = mask_low_counts(cube, threshold=100)
cube[~mask] = 0

# 3. Charge correction (C 1s example)
if region == "C1s":
    cube = charge_align_cube(cube, energy, ref_energy=284.8)

# 4. MCR decomposition
mcr_results = run_mcr_on_cube(cube, n_components=3)
plot_mcr_components(
    energy,
    mcr_results['component_spectra'],
    mcr_results['conc_maps'],
    mcr_results['method'],
    output_dir
)
```

### Pattern 3: PCA + Clustering for Phase Identification
```python
from case2_hyperspectral_processing import pca_cluster_analysis
from cluster_plots import plot_cluster_analysis, plot_cluster_spectra

# Run PCA and cluster pixels
cluster_results = pca_cluster_analysis(
    hyperspectral_map,
    n_pca=3,
    n_clusters=4
)

# Visualize results
plot_cluster_analysis(hyperspectral_map, cluster_results, output_dir)
plot_cluster_spectra(energy, cluster_results['cluster_info'], output_dir)
```

---

## Adding New Chemometrics Methods

### Example: Adding ICA (Independent Component Analysis)

1. **Add function to chemometrics_utils.py**:
```python
from sklearn.decomposition import FastICA

def run_ica_on_cube(cube: np.ndarray, n_components: int = 3) -> Dict:
    """
    Run ICA for blind source separation.
    Better for finding statistically independent sources.
    """
    m, n, p = cube.shape
    X = cube.reshape(m * n, p)
    
    ica = FastICA(n_components=n_components, random_state=0)
    S = ica.fit_transform(X)  # Sources (mixing coefficients)
    A = ica.components_      # Mixing matrix (pure spectra)
    
    source_maps = S.reshape(m, n, n_components)
    
    return {
        "method": "ICA",
        "sources": S,
        "source_maps": source_maps,
        "component_spectra": A.T
    }
```

2. **Add visualization to chemometrics_plots.py**:
```python
def plot_ica_components(energy, component_spectra, source_maps,
                       output_dir, base_name, region):
    """Plot ICA components similar to MCR."""
    # Similar structure to plot_mcr_components
    pass
```

3. **Integrate into workflow in XPS_map.py**:
```python
from chemometrics_utils import run_ica_on_cube
from chemometrics_plots import plot_ica_components

# In process_hyperspectral_map_simple():
if do_ica:
    ica_results = run_ica_on_cube(cube, n_components=n_ica)
    if make_plots:
        plot_ica_components(energy, ica_results['component_spectra'],
                           ica_results['source_maps'], output_dir,
                           map_data.name, map_data.metadata.region)
```

---

## Key Decisions for Modular Design

### Why separate chemometrics_utils and chemometrics_plots?
- **Reusability**: Utilities can be imported without matplotlib dependency
- **Testing**: Algorithms can be unit-tested independently
- **Batch processing**: Can run analysis without generating plots

### Why not merge case1 and case2 processing?
- **Fundamentally different data**: 2D vs 3D structures
- **Different algorithms**: Segmentation vs spectral decomposition
- **Different workflows**: Image processing vs chemometrics

### When to add to existing modules vs create new ones?
- **Add to existing**: Function fits existing theme (e.g., new clustering method → cluster_plots.py)
- **Create new module**: New functionality domain (e.g., time-series analysis → `temporal_analysis.py`)

---

## Dependencies Summary

### Required (Core functionality)
- numpy
- scipy
- scikit-learn (PCA, NMF, KMeans)
- matplotlib
- PyYAML

### Optional (Enhanced features)
- pymcr (MCR-ALS, fallback to NMF if unavailable)
- scikit-image (advanced morphological operations)

### Installation
```bash
# Core
pip install numpy scipy scikit-learn matplotlib pyyaml

# Optional
pip install pymcr scikit-image
```

---

## Performance Considerations

### Memory optimization for large maps
```python
# Use MiniBatchKMeans for >10,000 pixels
from sklearn.cluster import MiniBatchKMeans

km = MiniBatchKMeans(n_clusters=4, batch_size=2048)
```

### Parallel processing for batch analysis
```python
from multiprocessing import Pool

def process_file(file_path):
    # Process single map
    pass

# Process multiple maps in parallel
with Pool(4) as p:
    results = p.map(process_file, file_paths)
```

---

## Testing New Chemometrics Features

### Unit test template
```python
# tests/test_chemometrics_utils.py
import numpy as np
from chemometrics_utils import compute_pre_image

def test_pre_image():
    # Create synthetic data
    cube = np.random.rand(10, 10, 50)
    
    # Compute PRE
    pre = compute_pre_image(cube)
    
    # Assertions
    assert pre.shape == (10, 10)
    assert np.all(pre >= 0)  # Entropy is non-negative
    assert not np.any(np.isnan(pre))
```

---

## Future Enhancements

### Planned modules
1. **spatial_correlation.py** - Spatial autocorrelation, Moran's I
2. **temporal_analysis.py** - Time-series analysis for in-situ XPS
3. **multivariate_stats.py** - ANOVA, PLS-DA for classification
4. **export_utils.py** - Export to HDF5, NetCDF for interoperability

### Planned visualization
1. **interactive_plots.py** - Plotly/Bokeh interactive maps
2. **3d_visualization.py** - Volume rendering for depth profiling
3. **report_generator.py** - Automated PDF/HTML reports

---

## References

### Chemometrics methods
- PRE: H. Xu et al., *Surf. Interface Anal.* (2012)
- MCR-ALS: R. Tauler, *Chemometrics Intell. Lab. Syst.* (1995)
- Charge correction: J. F. Moulder et al., *Handbook of XPS* (1992)

### XPS-specific resources
- XPS spectral library: [NIST XPS Database](https://srdata.nist.gov/xps/)
- Data formats: ISO 14976 (VAMAS), ISO 19318 (PHI MultiPak)
