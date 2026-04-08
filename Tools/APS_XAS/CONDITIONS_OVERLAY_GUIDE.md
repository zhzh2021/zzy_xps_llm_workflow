# Experimental Conditions Overlay on PCA Space

## Overview

The `plot_conditions_overlay()` method creates a multi-panel visualization showing how different experimental conditions (pH, ligand type, iron source, concentrations, etc.) map onto the PCA space. This provides comprehensive insight into which experimental variables drive the principal components.

## What Was Added

### New Method in `xas_experiment_planner.py`

```python
def plot_conditions_overlay(
    self,
    pca_result,
    experimental_params: Dict[str, List[Any]],
    pc_x: int = 1,
    pc_y: int = 2,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[Path] = None
)
```

**Purpose**: Creates a multi-panel figure where each panel shows the same PCA space (PC1 vs PC2) colored/styled by a different experimental variable.

**Key Features**:
- **Automatic layout**: Calculates optimal grid layout (tries to make roughly square)
- **Smart coloring**: 
  - Continuous variables (many unique values) → continuous colormap (viridis)
  - Categorical variables (few unique values) → distinct colors with legend
- **Sample labels**: Each point labeled with sample name
- **Consistent axes**: All panels use same PC axes for easy comparison

## Usage Example

### Basic Usage

```python
from xas_ml_modules.xas_experiment_planner import XASExperimentPlanner
from xas_ml_modules.xas_spectrum_pca import XASSpectrumPCA

# 1. Run PCA first
analyzer = XASSpectrumPCA()
pca_result = analyzer.analyze_spectra(energies, spectra, sample_names)

# 2. Prepare experimental parameters
experimental_params = {
    'pH': [2.0, 2.0, 5.1, 5.2, 5.3, ...],  # pH values
    'ligand': ['Malic acid', 'Malic acid', 'Tartaric acid', ...],  # Categorical
    'iron_source': ['FeCl2', 'FeSO4', 'FeCl2', ...],  # Categorical  
    'ligand_concentration': [0.1, 0.1, 0.05, ...],  # Continuous
    'state': ['gel', 'solution', 'gel', ...]  # Categorical
}

# 3. Create overlay plot
planner = XASExperimentPlanner()
planner.plot_conditions_overlay(
    pca_result,
    experimental_params=experimental_params,
    pc_x=1,  # X-axis: PC1
    pc_y=2,  # Y-axis: PC2
    figsize=(18, 12),  # Figure size
    save_path='conditions_overlay.png'
)
```

### Integration with Full Workflow

```python
# In test_spectrum_pca_planning.py or similar:

# ... after running PCA and extracting experimental parameters ...

# Generate conditions overlay
planner.plot_conditions_overlay(
    pca_result,
    experimental_params=experimental_params,
    pc_x=1, pc_y=2,
    figsize=(16, 10),
    save_path=plots_dir / 'conditions_overlay.png'
)
```

## Experimental Parameters Format

**Dict structure**: `parameter_name -> list of values`

**Requirements**:
- Each list must have same length as number of samples
- Values in same order as `pca_result.sample_names`

**Example for 4 samples**:
```python
experimental_params = {
    'pH': [2.2, 5.1, 5.2, 5.3],
    'ligand': ['Malic', 'Malic', 'Tartaric', 'Tartaric'],
    'temperature': [25, 25, 30, 30],
    'concentration': [0.1, 0.1, 0.05, 0.1]
}
```

## Output Description

### Multi-Panel Layout

The method creates a grid of subplots:
- **Number of panels** = number of experimental parameters
- **Layout**: Automatically calculated to be roughly square
  - 3 parameters → 2×2 grid (1 empty)
  - 5 parameters → 2×3 grid (1 empty)
  - 6 parameters → 2×3 grid (all filled)

### Each Panel Shows:

1. **Same PCA scores** (PC1 vs PC2)
2. **Different coloring** based on experimental variable
3. **Sample labels** for traceability
4. **Grid lines** and axis crosshairs (x=0, y=0)
5. **Title**: "Colored by: [parameter_name]"
6. **Axes labels**: Include variance explained (e.g., "PC1 (82.1%)")

### Color Schemes:

- **Continuous variables**: Viridis colormap with colorbar
  - Example: pH, concentration, temperature
  - Gradient shows progression of values
  
- **Categorical variables**: Tab10 colors with legend
  - Example: ligand type, iron source, gel/solution
  - Distinct colors for each category

## Scientific Interpretation

### What to Look For:

1. **Gradient patterns**: If a continuous variable (like pH) shows a gradient from left→right or bottom→top, that variable correlates with that PC

2. **Cluster separation**: If categorical variable (like ligand type) shows clear spatial clustering, that variable drives PC variation

3. **No pattern**: If points are randomly colored, that variable doesn't strongly influence those PCs

### Example Insights:

**Scenario**: Your 42 Fe K-edge samples show:
- **pH panel**: Clear gradient along PC1 → pH drives major variation
- **Ligand panel**: Two distinct clusters → Malic vs Tartaric acid affects structure
- **Iron source panel**: Random distribution → FeCl2 vs FeSO4 less important
- **State panel**: Slight separation → Gel vs solution has minor effect

**Conclusion**: pH is primary driver (PC1), ligand type is secondary (PC2), anion and state are less critical

## File Output

**Location**: Specified by `save_path` parameter
**Format**: PNG (high resolution, 300 dpi)
**Size**: Typically 1-2 MB depending on number of panels

**Default paths in workflow**:
```
04_ml_analysis/analysis_plots/conditions_overlay.png
```

## Comparison with Other Plots

| Plot Type | Purpose | Shows |
|-----------|---------|-------|
| `plot_conditions_overlay()` | **Comprehensive variable mapping** | Multiple conditions side-by-side |
| `plot_experiment_planning()` | **Suggestion visualization** | One condition + suggestions |
| `plot_scores()` | **Basic PCA visualization** | Scores only, optional single color-by |
| `plot_loadings()` | **Component interpretation** | Which energy regions matter |

**Use overlay when**: You want to understand which experimental variables drive your PCA results

**Use planning plot when**: You want to visualize where new experiments should go

## Testing

Run the standalone test:
```bash
python test_conditions_overlay_only.py
```

This will:
1. Load 42 normalized XAS spectra
2. Run PCA (5 components, 7100-7200 eV)
3. Extract experimental parameters from file names
4. Generate conditions overlay plot with 5 panels

**Expected output**: `conditions_overlay.png` in analysis_plots/

## Requirements

- **matplotlib**: For plotting
- **numpy**: For numerical operations
- **scipy**: Optional (for convex hull in other methods)

## Customization

### Adjust Figure Size

```python
# Larger plot for presentations
planner.plot_conditions_overlay(
    ...,
    figsize=(20, 14)  # Width, height in inches
)

# Smaller plot for reports
planner.plot_conditions_overlay(
    ...,
    figsize=(12, 8)
)
```

### Select Different PCs

```python
# Compare PC2 vs PC3
planner.plot_conditions_overlay(
    ...,
    pc_x=2,  # PC2 on X-axis
    pc_y=3   # PC3 on Y-axis
)
```

### Subset of Parameters

```python
# Only show most important variables
key_params = {
    'pH': experimental_params['pH'],
    'ligand': experimental_params['ligand']
}

planner.plot_conditions_overlay(
    pca_result,
    experimental_params=key_params,  # Subset only
    ...
)
```

## Troubleshooting

### Issue: "No experimental parameters provided"
**Cause**: Empty `experimental_params` dict  
**Fix**: Ensure dict has at least one parameter

### Issue: ValueError about list lengths
**Cause**: Parameter list length ≠ number of samples  
**Fix**: Check all lists have same length as `pca_result.sample_names`

### Issue: All points same color
**Cause**: All values in parameter list are identical  
**Fix**: This is expected! Means no variation in that variable

### Issue: Plot shows but doesn't save
**Cause**: Invalid `save_path`  
**Fix**: Ensure directory exists, use Path object or valid string

## Best Practices

1. **Order parameters by importance**: Put most important variables first (they appear in top-left panels)

2. **Limit to 6-8 parameters**: Too many panels become hard to read

3. **Use descriptive parameter names**: They appear in panel titles

4. **Combine with other plots**: 
   - Use overlay for variable discovery
   - Use planning plot for experiment design
   - Use loadings plot for spectral interpretation

5. **Save high resolution**: Default 300 dpi is good for publications

## References

- **Module**: `xas_ml_modules/xas_experiment_planner.py`
- **Method**: Lines 710-780 (approx)
- **Test script**: `test_conditions_overlay_only.py`
- **Example output**: `04_ml_analysis/analysis_plots/conditions_overlay.png`

## Version History

- **v1.0** (2026-03-06): Initial implementation
  - Multi-panel layout
  - Automatic categorical vs continuous detection
  - Smart coloring and legends
  - Integration with experiment planner workflow

---

## Quick Reference

**Minimal example**:
```python
planner.plot_conditions_overlay(pca_result, experimental_params)
```

**Typical example**:
```python
planner.plot_conditions_overlay(
    pca_result, 
    experimental_params, 
    save_path='overlay.png'
)
```

**Full control**:
```python
planner.plot_conditions_overlay(
    pca_result,
    experimental_params,
    pc_x=1,
    pc_y=2,
    figsize=(18, 12),
    save_path=output_dir / 'conditions_overlay.png'
)
```
