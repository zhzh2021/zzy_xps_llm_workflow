# Whole-Spectrum PCA for XAS Analysis

**Module:** `xas_spectrum_pca.py`  
**Purpose:** Discover spectral structure without predefined features  
**Author:** ZZY Lab  
**Date:** March 5, 2026

---

## Overview

Whole-spectrum PCA performs Principal Component Analysis directly on XAS spectra (energy vs. μ(E)) rather than on extracted features. This approach discovers hidden patterns, reaction pathways, and spectral variations **without making assumptions** about which features are important.

### Why Whole-Spectrum PCA?

**Feature-based PCA** (existing `xas_pca_analyzer.py`):
- Requires feature extraction first (edge position, white line, etc.)
- Limited to predefined features
- May miss unexpected variations
- Good for: Known feature relationships, interpretability

**Whole-spectrum PCA** (this module):
- Analyzes entire spectral shape
- Discovers unexpected patterns
- Captures subtle variations
- Good for: Exploration, reaction monitoring, complex mixtures

---

## What It Captures

### 1. **Oxidation State Evolution**
```
Fe(II) → Fe(III) oxidation:
  PC1 ← Edge position shift (7112 → 7115 eV)
  PC2 ← White line intensity change
  PC3 ← Pre-edge feature evolution
```

### 2. **Coordination Environment Changes**
```
Octahedral ↔ Tetrahedral:
  PC1 ← Overall spectral shape
  PC2 ← XANES oscillation frequency
  PC3 ← White line position/width
```

### 3. **Reaction Pathways**
```
Multi-step reactions:
  Scores trajectory → Reaction progression
  PC loadings → Spectral features changing
```

### 4. **Hidden Spectral Variations**
```
Unexpected correlations:
  PC3, PC4, ... → Subtle effects
  (temperature, pH, ligand effects)
```

---

## Workflow

### Input
- **Raw XAS spectra**: Energy and μ(E) arrays
- **From xarray.Dataset**: Direct integration with APS reader
- **Multiple formats**: Transmission, fluorescence, reference

### Processing Steps

#### 1. **Normalize Spectra**
```python
Methods:
  'standard'  → Zero mean, unit variance (default)
  'minmax'    → Scale to [0, 1] per spectrum
  'none'      → No normalization
```

**Standard normalization** (recommended):
- Removes intensity variations
- Focuses on spectral shape
- Makes spectra comparable

#### 2. **Interpolate to Common Energy Grid**
```python
# Automatic:
- Find overlapping energy range
- Create uniform grid (default: 500 points)
- Linear interpolation

# Manual control:
energy_range = (7050, 7200)  # eV
n_grid_points = 400
```

#### 3. **Assemble Spectrum Matrix**
```
Matrix structure:
  Rows = Spectra (samples)
  Columns = Energy points
  
Example (5 samples, 400 points):
  [μ₁(E₁) μ₁(E₂) ... μ₁(E₄₀₀)]
  [μ₂(E₁) μ₂(E₂) ... μ₂(E₄₀₀)]
  [μ₃(E₁) μ₃(E₂) ... μ₃(E₄₀₀)]
  [μ₄(E₁) μ₄(E₂) ... μ₄(E₄₀₀)]
  [μ₅(E₁) μ₅(E₂) ... μ₅(E₄₀₀)]
```

#### 4. **Run PCA**
```python
# Eigenvalue decomposition:
  Spectra = Scores × Loadings + Residual
  
# Output:
  Scores (n_spectra × n_components)
  Loadings (n_components × n_energy_points)
  Variance explained
```

### Outputs

#### **Scores** → Sample Clustering & Trajectories
```python
scores.shape = (n_spectra, n_components)

# PC1 vs PC2 plot reveals:
- Sample groupings (clusters)
- Reaction trajectories (time series)
- Outliers (unusual spectra)
```

#### **Loadings** → Spectral Interpretation
```python
loadings.shape = (n_components, n_energy_points)

# Plot PC loadings shows:
- Which energy regions drive variation
- Positive/negative contributions
- Spectral features responsible for separation
```

#### **Variance Ratio** → Component Importance
```python
variance_ratio = [0.65, 0.22, 0.08, 0.03, ...]

# Interpretation:
PC1 = 65% → Dominant spectral variation
PC2 = 22% → Secondary variation
PC3-PCn → Subtle effects
```

---

## Usage Examples

### Example 1: Basic Analysis

```python
from xas_ml_modules.xas_spectrum_pca import XASSpectrumPCA
from aps_xas_reader import load_aps_dataset

# Load XAS data
datasets = load_aps_dataset('data_dir/', pattern='FeCl2*')

# Initialize analyzer
analyzer = XASSpectrumPCA(
    variance_threshold=0.95,  # Keep 95% variance
    normalization='standard',
    n_grid_points=400
)

# Run analysis
result = analyzer.analyze_datasets(
    datasets=datasets,
    mu_variable='mu_trans'
)

# View results
print(result.summary())
print(f"Scores shape: {result.scores.shape}")
print(f"Loadings shape: {result.loadings.shape}")
```

### Example 2: Reaction Monitoring

```python
# Time-series XAS data
time_points = [0, 5, 10, 20, 30, 60, 120]  # minutes
datasets = [load_data(f't{t}min.dat') for t in time_points]

# Analyze
result = analyzer.analyze_datasets(datasets)

# Plot reaction trajectory
analyzer.plot_scores(
    result,
    pc_x=1, pc_y=2,
    color_by=time_points,  # Color by reaction time
    save_path='reaction_trajectory.png'
)

# Scores trace shows reaction progression:
# PC1: Reactant → Product conversion
# PC2: Intermediate formation
```

### Example 3: pH Series Analysis

```python
# pH variation study
pH_values = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
datasets = [load_data(f'pH{pH}.dat') for pH in pH_values]
sample_names = [f'pH_{pH}' for pH in pH_values]

# Analyze
result = analyzer.analyze_spectra(
    energies=[ds['energy'].values for ds in datasets],
    spectra=[ds['mu_trans'].values for ds in datasets],
    sample_names=sample_names
)

# Identify pH-dependent features
analyzer.plot_loadings(result, components=[1, 2, 3])

# PC1 loading shows energy region most sensitive to pH
```

### Example 4: Custom Energy Range

```python
# Focus on XANES region only
analyzer = XASSpectrumPCA(
    energy_range=(7100, 7160),  # XANES only
    n_grid_points=300,
    normalization='standard'
)

result = analyzer.analyze_datasets(datasets)

# This reveals XANES-specific variations
# (ignoring pre-edge and extended regions)
```

---

## Interpretation Guide

### Reading Scores Plots

**PC1 vs PC2 scatter plot:**

```
      PC2
       ↑
   B B |     A A
   B   |   A A
  ─────┼─────────→ PC1
   C C | C C
       | C
```

- **Clusters**: Different chemical states (A, B, C)
- **Linear trajectory**: Progressive reaction (time/dose)
- **Outliers**: Unusual or contaminated samples
- **Spread**: Measurement reproducibility

### Reading Loadings Plots

**PC1 loading vs energy:**

```python
  Loading
    ↑
    |     /\        Edge shift
    |    /  \       contribution
    |   /    \_
  0 |──────────────
    |         /\    White line
    |        /  \   contribution
    ↓
    Energy (eV) →
```

- **Positive peak**: Features increasing in high-PC1 samples
- **Negative peak**: Features decreasing in high-PC1 samples
- **Zero crossing**: Energy where no variation occurs
- **Peak width**: Spectroscopic feature breadth

### Variance Interpretation

```python
PC1 = 68% → Main chemical variation (e.g., oxidation state)
PC2 = 18% → Secondary effect (e.g., coordination)
PC3 = 8%  → Subtle variation (e.g., ligand field)
PC4+ < 5% → Noise or very minor effects
```

**Rule of thumb:**
- Keep components until 95% cumulative variance
- Typically 2-4 components are meaningful
- More components with diverse sample sets

---

## Comparison: Feature PCA vs Whole-Spectrum PCA

| Aspect | Feature PCA | Whole-Spectrum PCA |
|--------|-------------|-------------------|
| **Input** | Extracted features (edge, white line, etc.) | Full μ(E) spectrum |
| **Dimensionality** | ~10-20 features | 200-500 energy points |
| **Interpretation** | Clear (known features) | Requires spectral knowledge |
| **Discovery** | Limited to defined features | Can find unexpected patterns |
| **Speed** | Fast (small matrix) | Slower (large matrix) |
| **Use case** | Known feature relationships | Exploratory, unsupervised |
| **Sensitivity** | Depends on feature extraction | Captures all spectral detail |

**When to use each:**

```python
# Use Feature PCA when:
- You know which features matter
- Want interpretable components
- Need fast processing
- Working with extracted features

# Use Whole-Spectrum PCA when:
- Exploring unknown systems
- Monitoring reactions
- Want maximum sensitivity
- Don't want to bias with features
```

---

## Advanced Features

### Auto-Component Selection

```python
# Method 1: Variance threshold
analyzer = XASSpectrumPCA(variance_threshold=0.95)
# → Keeps components until 95% variance captured

# Method 2: Fixed number
analyzer = XASSpectrumPCA(n_components=3)
# → Always keeps 3 components

# Method 3: Kaiser criterion (eigenvalue > 1)
# Automatically applied in results
print(result.kaiser_criterion)  # Number of significant components
```

### Normalization Methods

```python
# Standard normalization (default)
# Best for comparing spectral shapes
analyzer = XASSpectrumPCA(normalization='standard')

# Min-max normalization
# Useful when absolute intensity matters
analyzer = XASSpectrumPCA(normalization='minmax')

# No normalization
# When spectra already normalized
analyzer = XASSpectrumPCA(normalization='none')
```

### Integration with APS Reader

```python
# Direct xarray.Dataset input
from aps_xas_reader import load_aps_xas

datasets = [load_aps_xas(f) for f in files]

# Automatic extraction of:
# - Energy coordinate
# - mu_trans, mu_ref, or fluor_total
# - Sample names from metadata

result = analyzer.analyze_datasets(
    datasets=datasets,
    mu_variable='mu_trans'  # or 'fluor_total'
)
```

### Multiple Measurement Modes

```python
# Transmission mode
result_trans = analyzer.analyze_datasets(
    datasets, mu_variable='mu_trans'
)

# Fluorescence mode
result_fluor = analyzer.analyze_datasets(
    datasets, mu_variable='fluor_total'
)

# Reference foil
result_ref = analyzer.analyze_datasets(
    datasets, mu_variable='mu_ref'
)
```

---

## Visualization

### 1. Scree Plot
Shows variance explained by each component.

```python
analyzer.plot_scree(result, save_path='scree.png')
```

**Interpretation:**
- Steep drop-off → Few dominant components
- Gradual decline → Many components needed
- "Elbow" point → Optimal number of components

### 2. Scores Plot
Shows sample positions in PC space.

```python
analyzer.plot_scores(
    result,
    pc_x=1, pc_y=2,
    color_by=pH_values,           # Color by experimental variable
    save_path='scores.png'
)
```

**Applications:**
- Identify sample clusters
- Track reaction trajectories
- Spot outliers
- Visualize experimental conditions

### 3. Loadings Plot
Shows spectral features driving each PC.

```python
analyzer.plot_loadings(
    result,
    components=[1, 2, 3],         # Which PCs to plot
    save_path='loadings.png'
)
```

**Interpretation:**
- Peaks → Important energy regions
- Positive → Increasing in high scores
- Negative → Decreasing in high scores
- Compare to reference spectra for assignment

---

## Export & Integration

### Export Results

```python
analyzer.export_results(result, output_dir='pca_results/')

# Creates:
# - pca_scores.csv        → For clustering, plotting
# - pca_loadings.csv      → For spectral interpretation
# - pca_variance.csv      → Component importance
# - pca_summary.txt       → Text report
```

### CSV Format

**pca_scores.csv:**
```csv
sample,PC1,PC2,PC3
Sample_1,-2.456,0.832,-0.123
Sample_2,-1.987,1.045,0.234
Sample_3,0.567,-0.234,0.567
```

**pca_loadings.csv:**
```csv
energy_eV,PC1,PC2,PC3
7100.00,0.0234,-0.0456,0.0123
7100.50,0.0245,-0.0445,0.0134
7101.00,0.0267,-0.0423,0.0156
```

### Downstream Analysis

```python
import pandas as pd
import numpy as np

# Load scores for clustering
scores_df = pd.read_csv('pca_results/pca_scores.csv', index_col=0)

# k-means clustering on PCA scores
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(scores_df[['PC1', 'PC2']])

# Load loadings for interpretation
loadings_df = pd.read_csv('pca_results/pca_loadings.csv')
pc1_loading = loadings_df[['energy_eV', 'PC1']]

# Find key energies (max absolute loading)
important_energies = loadings_df.iloc[
    loadings_df['PC1'].abs().nlargest(10).index
]['energy_eV'].values
```

---

## Best Practices

### ✅ DO:

1. **Use sufficient samples**: Need at least 5-10 spectra for reliable PCA
2. **Check energy overlap**: All spectra should cover the same energy range
3. **Normalize appropriately**: Standard normalization for shape comparison
4. **Examine all components**: Don't just look at PC1-PC2
5. **Validate patterns**: Confirm PC trends match chemical expectations
6. **Plot loadings**: Always inspect loadings for spectral interpretation

### ❌ DON'T:

1. **Mix measurement modes**: Don't combine transmission + fluorescence
2. **Use too few points**: Grid should have 200+ points for detail
3. **Over-interpret noise**: Components with <2% variance may be noise
4. **Ignore quality**: Remove bad spectra before PCA
5. **Forget calibration**: Ensure energy scales are correct
6. **Mix edge types**: Don't combine Fe K-edge with Cu K-edge

---

## Troubleshooting

### Issue: "Need at least 2 spectra for PCA"
**Cause**: Insufficient samples  
**Solution**: Load more datasets

### Issue: PC1 captures >90% variance
**Cause**: One spectrum very different (outlier) or all spectra very similar  
**Solution**: Check data quality, remove outliers, or use more diverse samples

### Issue: Many components needed for 95% variance
**Cause**: Noisy data or very diverse sample set  
**Solution**: 
- Smooth spectra before PCA
- Check data quality
- Use fewer components if interpretable

### Issue: Negative loadings everywhere
**Cause**: Normal! PCA sign is arbitrary  
**Solution**: Flip sign if needed: `loading = -loading`

### Issue: Loadings look like pure noise
**Cause**: High-order PCs often are noise  
**Solution**: Focus on first 2-3 PCs, ignore rest

---

## Scientific Applications

### 1. Oxidation State Evolution

**Fe(II) → Fe(III) oxidation tracking:**
```python
# Time-series data during oxidation
times = [0, 1, 2, 5, 10, 20, 40, 60]  # minutes
datasets = [load_sample(f't{t}.dat') for t in times]

result = analyzer.analyze_datasets(datasets)

# PC1 captures edge shift (oxidation state)
# Scores increase with time → oxidation progress
# Loadings show edge energy shift
```

### 2. Coordination Chemistry

**Ligand exchange study:**
```python
# Different ligands with same metal
ligands = ['water', 'malic_acid', 'tartaric_acid', 'citric_acid']
datasets = [load_sample(f'Fe_{lig}.dat') for lig in ligands]

result = analyzer.analyze_datasets(datasets)

# PC1: Field strength differences
# PC2: Coordination number differences
# Clusters in scores → Ligand groups
```

### 3. Reaction Mechanism

**Multi-step reaction:**
```python
# Intermediate detection
# PC1: Reactant → Product
# PC2: Intermediate appearance/disappearance
# 3D trajectory (PC1, PC2, PC3) shows full mechanism
```

### 4. Sample Classification

**Unknown identification:**
```python
# Train on known samples
known_datasets = [...]
result_known = analyzer.analyze_datasets(known_datasets)

# Project unknown onto same PC space
unknown_datasets = [...]
result_unknown = analyzer.analyze_datasets(unknown_datasets)

# Compare scores to classify
```

---

## Module API Summary

### Classes

**`XASSpectrumPCA`**: Main analyzer class
- `__init__(n_components, variance_threshold, normalization, energy_range, n_grid_points)`
- `analyze_datasets(datasets, sample_names, mu_variable)` → SpectrumPCAResult
- `analyze_spectra(energies, spectra, sample_names)` → SpectrumPCAResult
- `plot_scree(result, save_path)`
- `plot_scores(result, pc_x, pc_y, color_by, save_path)`
- `plot_loadings(result, components, save_path)`
- `export_results(result, output_dir)`

**`SpectrumPCAResult`**: Results container
- Attributes: `n_components`, `scores`, `loadings`, `energy_grid`, `variance_ratio`, etc.
- Methods: `summary()`, `__repr__()`

---

## References & Further Reading

### PCA Background
- Jolliffe, I. T. "Principal Component Analysis" (Springer, 2002)
- Pearson, K. "On Lines and Planes of Closest Fit" (1901)

### XAS-Specific PCA
- Ressler, T. et al. "Bulk Structural Investigation of Mixed CuCo Oxides" *J. Catal.* 1997
- Manceau, A. et al. "Quantitative Zn Speciation in Smelter-Contaminated Soils" *Geochim. Cosmochim. Acta* 2000
- Calvin, S. "XAFS for Everyone" Chapter 5 (CRC Press, 2013)

### Multivariate Analysis
- Smolders, E. et al. "Internal Metal Sequestration" *Environ. Sci. Technol.* 2003

---

## Version History

**v1.0 (March 5, 2026)**
- Initial release
- xarray.Dataset integration
- Auto-component selection
- Comprehensive plotting
- CSV export functionality

---

**Module Location**: `APS_XAS/xas_ml_modules/xas_spectrum_pca.py`  
**Demo Script**: `APS_XAS/demo_whole_spectrum_pca.py`  
**Requirements**: `numpy`, `scikit-learn`, `xarray`, `matplotlib` (optional)

---

*For questions or contributions, contact ZZY Lab.*
