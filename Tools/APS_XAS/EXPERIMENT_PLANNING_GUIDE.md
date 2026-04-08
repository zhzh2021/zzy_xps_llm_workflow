# XAS Experiment Planning Guide

**Using PCA to Guide Experimental Design**

Version: 1.0  
Date: March 5, 2026  
Module: `xas_ml_modules.xas_experiment_planner`

---

## Table of Contents

1. [Overview](#overview)
2. [Concept](#concept) 
3. [Workflow](#workflow)
4. [API Reference](#api-reference)
5. [Strategy Comparison](#strategy-comparison)
6. [Interpretation Guide](#interpretation-guide)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

---

## Overview

The **XAS Experiment Planner** uses whole-spectrum PCA to guide the design of next experiments. Instead of random or grid-based sampling, this approach:

1. **Interprets PCA axes** in physical/chemical terms
2. **Maps experimental conditions** onto PCA space
3. **Identifies unexplored regions** using convex hull
4. **Suggests optimal experiments** to maximize information gain

This enables **intelligent experiment design** that efficiently explores chemical space.

---

## Concept

### PCA as a Chemical Coordinate System

After running whole-spectrum PCA on your XAS data:

```
Original space: μ(E) for each spectrum (high-dimensional)
PCA space: PC1, PC2, PC3, ... (low-dimensional)
```

**Each PC represents a chemical variation:**

| PC | Spectral Region | Physical Interpretation |
|----|----------------|------------------------|
| PC1 | Edge shift | Oxidation state changes |
| PC2 | White line | Coordination geometry |
| PC3 | EXAFS oscillations | Bond distances |

### Experimental Parameters on PCA Space

Plot your experimental conditions (pH, temperature, potential, composition) on the PCA space:

```
pH 2.2 → (PC1 = -11.07, PC2 = -0.00)
pH 5.1 → (PC1 = +3.69, PC2 = -0.02)
pH 5.2 → (PC1 = +3.70, PC2 = -0.02)
pH 5.3 → (PC1 = +3.68, PC2 = +0.04)
```

### Information Gain Strategies

**Empty regions = Unexplored chemistry**

Four strategies to maximize information:

1. **MaxDist**: Farthest from all existing points
2. **Boundary**: Between different clusters (transition states)
3. **Trajectory**: Fill gaps in time/reaction series
4. **Hull**: Expand convex hull (exploration frontier)

---

## Workflow

### Step 1: Run PCA

```python
from xas_ml_modules.xas_spectrum_pca import XASSpectrumPCA

analyzer = XASSpectrumPCA()
pca_result = analyzer.analyze_datasets(datasets, sample_names)
```

### Step 2: Initialize Planner

```python
from xas_ml_modules.xas_experiment_planner import XASExperimentPlanner

planner = XASExperimentPlanner(
    edge_energy=7112.0,  # Fe K-edge
    xanes_range=(7100, 7160),
    exafs_range=(7160, 7500)
)
```

### Step 3: Interpret PCA Components

```python
interpretations = planner.interpret_components(
    pca_result,
    n_components=3
)

for interp in interpretations:
    print(f"{interp.pc_number}: {interp.interpretation}")
    print(f"  Dominant regions: {interp.peak_regions}")
    print(f"  Peak energies: {interp.peak_energies}")
```

**Example Output:**
```
PC1: Major edge position / oxidation state changes
  Dominant regions: ['edge', 'XANES']
  Peak energies: [7115.2, 7129.4]

PC2: Moderate white line / coordination geometry
  Dominant regions: ['XANES']
  Peak energies: [7130.5]
```

### Step 4: Map Experimental Conditions

```python
experimental_params = {
    'pH': [2.2, 5.1, 5.2, 5.3],
    'temperature': [25, 25, 25, 25],
    'ligand': ['malic acid'] * 4
}

mapping = planner.map_conditions_to_pca(
    pca_result,
    experimental_params
)
```

### Step 5: Identify Explored Region

```python
hull_points, hull = planner.identify_explored_region(
    pca_result,
    pc_x=1,  # PC1 on X-axis
    pc_y=2   # PC2 on Y-axis
)

print(f"Convex hull: {len(hull_points)} vertices")
print(f"PC1 range: [{pca_result.scores[:, 0].min():.2f}, {pca_result.scores[:, 0].max():.2f}]")
```

### Step 6: Suggest Next Experiments

```python
# Try multiple strategies
strategies = ['maxdist', 'boundary', 'hull']

for strategy in strategies:
    suggestions = planner.suggest_experiments(
        pca_result,
        experimental_params=experimental_params,
        strategy=strategy,
        n_suggestions=3,
        pc_x=1,
        pc_y=2
    )
    
    print(f"\n{strategy.upper()} Strategy:")
    for i, sug in enumerate(suggestions):
        print(f"  Suggestion {i+1}:")
        print(f"    PC1={sug.predicted_scores[0]:.2f}, PC2={sug.predicted_scores[1]:.2f}")
        print(f"    Distance to nearest: {sug.distance_to_nearest:.2f}")
        print(f"    Priority: {sug.priority:.2f}")
        print(f"    Reason: {sug.reason}")
```

### Step 7: Visualize

```python
planner.plot_experiment_planning(
    pca_result,
    experimental_params=experimental_params,
    suggestions=suggestions,
    pc_x=1,
    pc_y=2,
    color_by='pH',  # Color points by pH
    save_path='experiment_planning.png'
)
```

---

## API Reference

### `XASExperimentPlanner`

Main class for experiment planning.

#### Constructor

```python
XASExperimentPlanner(
    edge_energy: float = 7112.0,
    xanes_range: Tuple[float, float] = (7100, 7160),
    exafs_range: Tuple[float, float] = (7160, 7500)
)
```

**Parameters:**
- `edge_energy`: Approximate edge energy (eV) for region classification
- `xanes_range`: XANES region energy range (eV)
- `exafs_range`: EXAFS region energy range (eV)

#### Methods

##### `interpret_components()`

Interpret principal components in physical/chemical terms.

```python
interpret_components(
    pca_result: SpectrumPCAResult,
    n_components: Optional[int] = None,
    peak_threshold: float = 0.1
) -> List[PCInterpretation]
```

**Parameters:**
- `pca_result`: PCA results from `XASSpectrumPCA`
- `n_components`: Number of components to interpret (None = all)
- `peak_threshold`: Threshold for peak detection (fraction of max loading)

**Returns:** List of `PCInterpretation` objects with:
- `pc_number`: Component number (1-indexed)
- `variance_explained`: Fraction of variance
- `peak_energies`: Energies of dominant features
- `peak_regions`: Spectral regions ('pre-edge', 'edge', 'XANES', 'EXAFS')
- `interpretation`: Human-readable interpretation

---

##### `map_conditions_to_pca()`

Map experimental conditions to PCA space.

```python
map_conditions_to_pca(
    pca_result: SpectrumPCAResult,
    experimental_params: Dict[str, List[Any]],
    sample_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]
```

**Parameters:**
- `pca_result`: PCA results
- `experimental_params`: Dict of parameter_name → values
  - Example: `{'pH': [2.0, 5.1], 'temp': [25, 25]}`
- `sample_names`: Optional sample names (default: from pca_result)

**Returns:** Dictionary with PCA scores and parameters

---

##### `identify_explored_region()`

Identify explored region using convex hull.

```python
identify_explored_region(
    pca_result: SpectrumPCAResult,
    pc_x: int = 1,
    pc_y: int = 2
) -> Tuple[np.ndarray, Optional[ConvexHull]]
```

**Parameters:**
- `pca_result`: PCA results
- `pc_x`: X-axis PC (1-indexed)
- `pc_y`: Y-axis PC (1-indexed)

**Returns:**
- `hull_points`: Vertices of convex hull (Nx2 array)
- `hull_object`: scipy ConvexHull object (or None if unavailable)

**Note:** Requires scipy. Falls back to bounding box if scipy not installed.

---

##### `suggest_experiments()`

Suggest next experiments to maximize information gain.

```python
suggest_experiments(
    pca_result: SpectrumPCAResult,
    experimental_params: Optional[Dict[str, List[Any]]] = None,
    strategy: str = 'maxdist',
    n_suggestions: int = 3,
    pc_x: int = 1,
    pc_y: int = 2
) -> List[ExperimentSuggestion]
```

**Parameters:**
- `pca_result`: PCA results
- `experimental_params`: Optional experimental conditions (required for 'trajectory')
- `strategy`: Suggestion strategy
  - `'maxdist'`: Maximize distance from existing points
  - `'boundary'`: Sample cluster boundaries
  - `'trajectory'`: Fill gaps in time series
  - `'hull'`: Expand convex hull
- `n_suggestions`: Number of suggestions to generate
- `pc_x`, `pc_y`: PCs to use for 2D analysis

**Returns:** List of `ExperimentSuggestion` objects with:
- `strategy`: Strategy used
- `predicted_scores`: Predicted PCA scores
- `distance_to_nearest`: Distance to nearest existing point
- `reason`: Human-readable explanation
- `suggested_conditions`: Suggested experimental conditions (if available)
- `priority`: Priority score (higher = better)

---

##### `plot_experiment_planning()`

Create visualization of experiment planning.

```python
plot_experiment_planning(
    pca_result: SpectrumPCAResult,
    experimental_params: Optional[Dict[str, List[Any]]] = None,
    suggestions: Optional[List[ExperimentSuggestion]] = None,
    pc_x: int = 1,
    pc_y: int = 2,
    color_by: Optional[str] = None,
    save_path: Optional[Path] = None
)
```

**Parameters:**
- `pca_result`: PCA results
- `experimental_params`: Experimental conditions
- `suggestions`: Experiment suggestions to plot
- `pc_x`, `pc_y`: PCs for axes
- `color_by`: Parameter name to color points by
- `save_path`: Optional path to save figure

**Visualization includes:**
- Existing experiments (circles, color-coded)
- Convex hull (shaded blue region)
- Suggested experiments (yellow stars, sized by priority)
- Sample labels

---

## Strategy Comparison

| Strategy | Best For | Explores | Priority Metric |
|----------|----------|----------|-----------------|
| **MaxDist** | Global exploration | Regions far from all data | Distance to nearest |
| **Boundary** | Transition states | Between different clusters | Inter-cluster distance |
| **Trajectory** | Kinetics/mechanisms | Gaps in time series | Gap size |
| **Hull** | Frontier expansion | Just beyond explored region | Hull expansion |

### When to Use Each Strategy

#### MaxDist
- ✅ Initial exploration of chemical space
- ✅ Finding unexpected phenomena
- ✅ Broad surveys
- ❌ May suggest unrealistic conditions

**Example:** You've tested pH 2 and pH 5. MaxDist suggests pH 8, pH 10, etc.

#### Boundary
- ✅ Understanding transitions (acid/base, oxidized/reduced)
- ✅ Phase boundaries
- ✅ Intermediates
- ❌ Requires distinct clusters

**Example:** Clear separation between pH 2 (reduced) and pH 5 (oxidized). Boundary suggests pH 3-4.

#### Trajectory
- ✅ Time-series experiments (reactions, aging)
- ✅ Smooth parameter sweeps
- ✅ Mechanism studies
- ❌ Requires sequential data with time/sequence parameter

**Example:** Reaction sampled at t=0, 10, 100 min. Trajectory suggests t=2, 50 min.

#### Hull
- ✅ Systematic expansion
- ✅ Controlled exploration
- ✅ Balanced coverage
- ❌ May not explore far from data

**Example:** Extends just beyond current data boundary in all directions.

---

## Interpretation Guide

### Reading PC Loadings

**PC loading = how much each energy point contributes to that component**

Example PC1 loading:

```
Energy (eV) | Loading
7112        | +0.8    ← Strong positive peak at edge
7130        | +0.5    ← Moderate peak at white line
7200        | -0.3    ← Negative in EXAFS
```

**Interpretation:**
- Samples with high PC1 scores: higher edge energy, stronger white line
- Samples with low PC1 scores: lower edge energy, weaker white line
- **Physical meaning:** PC1 captures oxidation state (Fe²⁺ → Fe³⁺)

### Region Classification

Spectral regions are classified based on energy:

| Region | Energy Range | Features | Chemical Information |
|--------|--------------|----------|---------------------|
| **Pre-edge** | E < E₀ - 10 eV | 1s → 3d transitions | Symmetry, ligand field |
| **Edge** | E₀ - 10 to E₀ + 5 eV | Edge position | Oxidation state |
| **XANES** | E₀ + 5 to ~50 eV | White line, shape | Coordination, geometry |
| **EXAFS** | > E₀ + 50 eV | Oscillations | Bond distances, CN |

###Loading Patterns → Chemistry

| Loading Pattern | Physical Meaning | Example |
|----------------|------------------|---------|
| Peak at edge | Oxidation state change | Fe²⁺ ↔ Fe³⁺ |
| Peak at white line | Coordination change | 4-coord ↔ 6-coord |
| Oscillation in EXAFS | Bond length change | Fe-O distance |
| Multiple peaks | Complex chemistry | Multiple effects |

---

## Best Practices

### 1. Run Multiple Strategies

Don't rely on one strategy. Compare:

```python
strategies = ['maxdist', 'boundary', 'hull']
all_suggestions = {}

for strategy in strategies:
    all_suggestions[strategy] = planner.suggest_experiments(
        pca_result, 
        experimental_params=params,
        strategy=strategy
    )
```

**Then rank by priority across all strategies.**

### 2. Validate Suggestions

**Not all mathematical suggestions are chemically realistic:**

- Check if suggested conditions are experimentally accessible
- Consider stability (e.g., pH limits for certain minerals)
- Factor in practical constraints (temperature, pressure)

### 3. Start with Clusters

If you see distinct clusters:

1. Use **boundary** strategy first (understand transitions)
2. Then use **maxdist** or **hull** (expand coverage)

### 4. Use Domain Knowledge

PCA finds mathematical patterns. You know the chemistry.

**Combine both:**
- PC interpretation → hypothesis
- Domain knowledge → validation
- Suggested experiment → test

### 5. Iterate

Experiment planning is iterative:

```
1. Run initial experiments
2. Analyze with PCA
3. Suggest next experiments
4. Run suggested experiments
5. Re-analyze (now more spectra)
6. Update suggestions
7. Repeat
```

Each cycle improves your understanding.

### 6. Track Variance

If PC1 explains > 95% variance:
- **1D problem**: Focus on PC1 axis
- Use pH, potential, or single parameter
- Boundary strategy especially useful

If variance spread across PC1, PC2, PC3:
- **Multi-dimensional problem**: Need 3D exploration
- Use maxdist or hull strategies
- Consider multi-parameter experiments

---

## Examples

### Example 1: pH Series

**Data:** Fe spectra at pH 2.2, 5.1, 5.2, 5.3

**PCA Results:**
- PC1 explains 100% variance
- PC1 loadings peak at edge (7112 eV)
- Interpretation: Oxidation state change with pH

**Experiment Planning:**

```python
suggestions = planner.suggest_experiments(
    pca_result,
    experimental_params={'pH': [2.2, 5.1, 5.2, 5.3]},
    strategy='boundary'
)
```

**Top Suggestion:**
- PC1 = -3.68 (midpoint between pH 2.2 and pH 5.1)
- Estimated pH ≈ 3.7
- Reason: Explores transition between reduced (pH 2) and oxidized (pH 5) states

**Action:** Run experiment at pH 3.5-4.0 to capture oxidation transition.

---

### Example 2: Reaction Time Series

**Data:** Spectra at t = 0, 10, 100, 1000 minutes

**PCA Results:**
- PC1: 85% variance (reaction progress)
- PC2: 12% variance (side reactions)
- Clear trajectory in PC1-PC2 space

**Experiment Planning:**

```python
suggestions = planner.suggest_experiments(
    pca_result,
    experimental_params={'time': [0, 10, 100, 1000]},
    strategy='trajectory'
)
```

**Top Suggestions:**
- t ≈ 2 min (gap between 0 and 10)
- t ≈ 50 min (gap between 10 and 100)
- Reason: Large spectral changes occur in these gaps

**Action:** Add samples at t = 2, 50 min to resolve reaction mechanism.

---

### Example 3: Multi-Parameter Space

**Data:** Fe-ligand complexes
- 2 ligands (malic, tartaric acid)
- 2 salts (FeCl₂, FeSO₄)
- 3 pHvalues (2, 3, 5)
- 2 × 2 × 3 = 12 combinations (but only tested 6)

**PCA Results:**
- PC1: 60% variance (pH effect)
- PC2: 30% variance (ligand effect)
- PC3: 8% variance (salt effect)

**Experiment Planning:**

```python
suggestions = planner.suggest_experiments(
    pca_result,
    strategy='hull'
)
```

**Top Suggestions:**
- Conditions expanding convex hull in PC1-PC2 space
- Example: FeCl₂ + tartaric acid + pH 4 (not yet tested)

**Action:** Systematically fill gaps in parameter space guided by PCA.

---

## Advanced Topics

### Multi-Objective Optimization

Combine multiple criteria:

```python
def score_suggestion(sug, weights={'distance': 0.5, 'priority': 0.5}):
    return (
        weights['distance'] * sug.distance_to_nearest +
        weights['priority'] * sug.priority
    )

# Rank all suggestions
all_suggestions = []
for strategy in ['maxdist', 'boundary', 'hull']:
    all_suggestions.extend(
        planner.suggest_experiments(pca_result, strategy=strategy)
    )

all_suggestions.sort(key=score_suggestion, reverse=True)
best = all_suggestions[0]
```

### Constrained Optimization

Add experimental constraints:

```python
def is_feasible(sug, experimental_params):
    # Estimate pH from PC1
    estimated_ph = estimate_ph_from_pc1(sug.predicted_scores[0])
    
    # Check constraints
    if estimated_ph < 1 or estimated_ph > 12:
        return False
    
    # Other constraints...
    return True

feasible_suggestions = [sug for sug in suggestions if is_feasible(sug, params)]
```

### Active Learning Integration

Use experiment planner in active learning loop:

```python
for iteration in range(10):
    # 1. Run experiments
    new_spectra = run_experiments(current_conditions)
    
    # 2. Update dataset
    all_spectra.extend(new_spectra)
    
    # 3. Re-run PCA
    pca_result = analyzer.analyze_datasets(all_spectra)
    
    # 4. Suggest next experiments
    suggestions = planner.suggest_experiments(pca_result)
    
    # 5. Select best suggestion
    next_experiment = suggestions[0]
    current_conditions = next_experiment.suggested_conditions
```

---

## Summary

The XAS Experiment Planner transforms PCA from a **data analysis tool** into an **experimental design tool**.

**Key Capabilities:**
✅ Interpret PCA components physically  
✅ Map experimental space onto PCA space  
✅ Identify unexplored regions  
✅ Suggest experiments maximizing information gain  
✅ Multiple strategies for different scenarios  

**Best Practices:**
- Run multiple strategies, combine results
- Validate suggestions with domain knowledge
- Iterate: experiment → analyze → suggest → repeat
- Use appropriate strategy for your experiment type

**Next Steps:**
1. Run PCA on your XAS data
2. Interpret components (edge, XANES, EXAFS)
3. Generate suggestions with multiple strategies
4. Prioritize based on feasibility and scientific goals
5. Run suggested experiments
6. Repeat with expanded dataset

**The goal:** Efficient exploration → faster discovery.

---

## References

- Whole-Spectrum PCA Guide: `WHOLE_SPECTRUM_PCA_GUIDE.md`
- APS Reader Documentation: `APS_XAS_READER_DOCUMENTATION.md`
- Module code: `xas_ml_modules/xas_experiment_planner.py`
- Test script: `test_experiment_planning.py`

For questions or issues, consult the module docstrings or create an issue.

---

**Version History:**
- 1.0 (2026-03-05): Initial release
