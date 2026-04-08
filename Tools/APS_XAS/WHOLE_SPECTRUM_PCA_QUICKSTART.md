# Whole-Spectrum PCA Module - Quick Reference

## What Was Created

✅ **Module**: [`xas_spectrum_pca.py`](c:\Users\b82797\Documents\Github\zz_llm\zzy_llm\Tools\APS_XAS\xas_ml_modules\xas_spectrum_pca.py)  
✅ **Demo Script**: [`demo_whole_spectrum_pca.py`](c:\Users\b82797\Documents\Github\zz_llm\zzy_llm\Tools\APS_XAS\demo_whole_spectrum_pca.py)  
✅ **Documentation**: [`WHOLE_SPECTRUM_PCA_GUIDE.md`](c:\Users\b82797\Documents\Github\zz_llm\zzy_llm\Tools\APS_XAS\WHOLE_SPECTRUM_PCA_GUIDE.md)  
✅ **Updated**: Module `__init__.py` to export new classes  

---

## Key Concept

**Whole-spectrum PCA vs Feature-based PCA:**

```
Feature PCA:              Whole-Spectrum PCA:
Extract features    →     Work with full μ(E) spectrum
  ↓                         ↓
[e0, white_line, ...]     [μ(E₁), μ(E₂), ..., μ(Eₙ)]
  ↓                         ↓
PCA on ~15 features       PCA on ~400 energy points
  ↓                         ↓
Limited to defined        Discovers ALL spectral
features                  variations
```

**Captures without predefined features:**
- 🔬 Oxidation state evolution
- 🧪 Coordination environment changes  
- 🔄 Reaction pathways
- ✨ Hidden spectral variations

---

## Quick Start

### 1. Basic Usage with APS Data

```python
from xas_ml_modules.xas_spectrum_pca import XASSpectrumPCA
from aps_xas_reader import load_aps_dataset

# Load XAS spectra
datasets = load_aps_dataset('data_dir/', pattern='FeCl2*')

# Initialize analyzer
analyzer = XASSpectrumPCA(variance_threshold=0.95)

# Run whole-spectrum PCA
result = analyzer.analyze_datasets(datasets, mu_variable='mu_trans')

# View results
print(result.summary())
```

### 2. The Workflow

```
Step 1: Normalize spectra
  ↓  (standard normalization: zero mean, unit variance)
  
Step 2: Interpolate to common energy grid
  ↓  (automatic overlap detection, 400-500 points)
  
Step 3: Assemble spectrum matrix
  ↓  [n_spectra × n_energy_points]
  
Step 4: Run PCA
  ↓  Eigenvalue decomposition
  
Output:
  - Scores [n_spectra × n_components]    → clustering/trajectories
  - Loadings [n_components × n_energy]   → spectral interpretation
  - Variance ratio                       → component importance
```

---

## Understanding the Output

### Scores → Sample Clustering & Trajectories

```python
result.scores.shape  # (n_spectra, n_components)

# Example: 10 samples, 3 components
array([[ 2.177, -0.469,  0.238],   # Sample 1 position in PC space
       [ 1.740, -0.158, -0.342],   # Sample 2
       [ 1.317,  0.014, -0.141],   # Sample 3
       ...])

# Plot PC1 vs PC2 shows:
analyzer.plot_scores(result, pc_x=1, pc_y=2)
# → Clusters = different chemical states
# → Trajectories = reaction progression
# → Outliers = unusual samples
```

### Loadings → Spectral Interpretation

```python
result.loadings.shape  # (n_components, n_energy_points)

# PC1 loading shows which energies drive main variation
# Positive peak at 7115 eV → edge shift
# Negative peak at 7130 eV → white line change

analyzer.plot_loadings(result, components=[1, 2, 3])
# → Identify energy regions responsible for separation
# → Compare to reference spectra for assignment
```

### Variance Ratio → Component Importance

```python
result.variance_ratio
# array([0.881, 0.039, 0.014, 0.013, 0.012])

# PC1 = 88.1% → Main spectral variation
# PC2 = 3.9%  → Secondary effect
# PC3-PC5    → Minor variations

result.cumulative_variance
# array([0.881, 0.920, 0.934, 0.948, 0.959])
# 95.9% total variance captured by 5 components
```

---

## Visualization

### 1. Scree Plot
Shows how many components are needed.

```python
analyzer.plot_scree(result, save_path='scree.png')
```

### 2. Scores Plot  
Shows sample positions and clustering.

```python
analyzer.plot_scores(
    result,
    pc_x=1, pc_y=2,
    color_by=pH_values,  # Color by experimental variable
    save_path='scores.png'
)
```

### 3. Loadings Plot
Shows spectral features driving each PC.

```python
analyzer.plot_loadings(
    result,
    components=[1, 2, 3],
    save_path='loadings.png'
)
```

---

## Export Results

```python
analyzer.export_results(result, output_dir='pca_results/')

# Creates:
# pca_results/
#   ├── pca_scores.csv      ← Use for clustering
#   ├── pca_loadings.csv    ← Use for interpretation
#   ├── pca_variance.csv    ← Component importance
#   └── pca_summary.txt     ← Text report
```

---

## Configuration Options

```python
analyzer = XASSpectrumPCA(
    n_components=None,            # Auto-select (or specify: 3, 4, etc.)
    variance_threshold=0.95,      # Keep 95% variance
    normalization='standard',     # 'standard', 'minmax', or 'none'
    energy_range=None,            # Auto-detect (or specify: (7100, 7200))
    n_grid_points=400             # Common grid resolution
)
```

**Normalization methods:**
- `'standard'`: Zero mean, unit variance (default, **recommended**)
- `'minmax'`: Scale each spectrum to [0, 1]
- `'none'`: No normalization

---

## Example Applications

### Reaction Monitoring

```python
# Time-series XAS during oxidation
times = [0, 5, 10, 20, 30, 60, 120]  # minutes
datasets = [load_sample(f't{t}min.dat') for t in times]

result = analyzer.analyze_datasets(datasets)

# Plot trajectory colored by time
analyzer.plot_scores(result, color_by=times)
# Shows reaction progression in PC space
```

### pH Series Analysis

```python
# Different pH conditions
pH_values = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
datasets = [load_sample(f'pH{pH}.dat') for pH in pH_values]

result = analyzer.analyze_datasets(datasets)

# PC loadings show pH-sensitive energy regions
analyzer.plot_loadings(result)
```

### Sample Classification

```python
# Compare unknown to known samples
known_datasets = [...]
unknown_datasets = [...]

# Combine and analyze
all_datasets = known_datasets + unknown_datasets
result = analyzer.analyze_datasets(all_datasets)

# Check where unknowns fall in PC space
analyzer.plot_scores(result)
# Proximity to knowns → Classification
```

---

## API Reference

### Main Class

```python
class XASSpectrumPCA:
    def analyze_datasets(datasets, sample_names=None, mu_variable='mu_trans')
        → SpectrumPCAResult
    
    def analyze_spectra(energies, spectra, sample_names)
        → SpectrumPCAResult
    
    def plot_scree(result, save_path=None)
    def plot_scores(result, pc_x=1, pc_y=2, color_by=None, save_path=None)
    def plot_loadings(result, components=None, save_path=None)
    def export_results(result, output_dir)
```

### Result Object

```python
class SpectrumPCAResult:
    # Dimensions
    n_components: int
    n_spectra: int
    n_energy_points: int
    
    # Data
    energy_grid: np.ndarray
    spectra_matrix: np.ndarray
    scores: np.ndarray              # (n_spectra, n_components)
    loadings: np.ndarray            # (n_components, n_energy_points)
    
    # Variance
    explained_variance: np.ndarray
    variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    
    # Metadata
    sample_names: List[str]
    normalization_method: str
    energy_range: Tuple[float, float]
    confidence: float
    flags: List[str]
    
    # Methods
    def summary() → str
```

---

## When to Use This Module

### ✅ Use Whole-Spectrum PCA When:

- Exploring **unknown systems**
- Monitoring **reactions over time**
- Want **maximum sensitivity** to spectral changes
- Don't want to **bias** analysis with predefined features
- Looking for **unexpected patterns**
- Have **diverse sample sets**

### ❌ Use Feature PCA Instead When:

- You know **which features matter**
- Want **interpretable** components
- Need **fast processing**
- Working with **extracted features** already
- Have **small datasets** (< 5 samples)

---

## Testing

### Run Built-in Demo

```bash
# Test with synthetic data
python xas_ml_modules/xas_spectrum_pca.py

# Demo with real data
python demo_whole_spectrum_pca.py
```

### Expected Output

```
================================================================================
WHOLE-SPECTRUM PCA ANALYSIS RESULTS
================================================================================
Number of spectra: 10
Energy grid: 250 points (7000.1 - 7199.9 eV)
Normalization: standard (zero mean, unit variance)

Principal Components: 5
Total variance captured: 95.9%

Variance explained by each component:
  PC1: 88.1% (cumulative: 88.1%)
  PC2: 3.9% (cumulative: 92.0%)
  ...
✓ Module working correctly!
```

---

## Troubleshooting

**Issue**: "Need at least 2 spectra for PCA"  
**Solution**: Load more datasets

**Issue**: PC1 captures >95% variance  
**Solution**: Normal for homogeneous samples or one outlier present

**Issue**: Loadings look noisy  
**Solution**: Focus on PC1-PC3, higher PCs often just noise

**Issue**: Energy grids don't overlap  
**Solution**: Check energy ranges, ensure all spectra cover same region

---

## Files Created

```
APS_XAS/
├── xas_ml_modules/
│   ├── xas_spectrum_pca.py           ← Main module (NEW)
│   └── __init__.py                    ← Updated to export new classes
├── demo_whole_spectrum_pca.py         ← Demo script (NEW)
└── WHOLE_SPECTRUM_PCA_GUIDE.md        ← Full documentation (NEW)
```

---

## Next Steps

1. **Try the demo**: `python demo_whole_spectrum_pca.py`
2. **Load your data**: Use with APS XAS datasets
3. **Interpret results**: Check scores, loadings, variance
4. **Export for analysis**: Save CSVs for further processing
5. **Compare with feature PCA**: See which reveals more patterns

---

## Key Advantages

✨ **No feature extraction required** → Direct spectral analysis  
✨ **Discovers unexpected patterns** → Not limited to predefined features  
✨ **Captures full spectral detail** → Maximum information content  
✨ **Easy visualization** → Scores and loadings plots  
✨ **Integrates with APS reader** → Works with xarray.Dataset objects  
✨ **Auto-component selection** → Variance-based optimization  
✨ **Comprehensive export** → CSV files for downstream analysis  

---

**Module tested and ready to use!** 🎉

For detailed explanations, see [`WHOLE_SPECTRUM_PCA_GUIDE.md`](c:\Users\b82797\Documents\Github\zz_llm\zzy_llm\Tools\APS_XAS\WHOLE_SPECTRUM_PCA_GUIDE.md)
