# XAS Experiment Planning - Quick Start

**From Data to Next Experiment in 5 Minutes**

---

## The Big Picture

```
┌─────────────┐        ┌──────────────┐        ┌─────────────────┐
│   XAS Data  │   →    │ Whole-Spectrum│   →   │   Experiment    │
│  (APS HDF5/ │        │      PCA      │        │    Planning     │
│    ASCII)   │        │               │        │                 │
└─────────────┘        └──────────────┘        └─────────────────┘
     Load                Analyze               Suggest Next
```

**Goal:** Use PCA to discover patterns in XAS spectra and intelligently suggest next experiments.

---

## Quick Start (Copy-Paste Ready)

### 1. Load Your Data

```python
from pathlib import Path
from xas_reader.aps_xas_reader import load_aps_xas

# Load APS data (auto-detects ASCII or HDF5)
data_dir = Path("your/data/directory")
datasets = []
sample_names = []

for file in data_dir.glob("*.dat"):  # or other extensions
    dataset = load_aps_xas(file)
    datasets.append(dataset)
    sample_names.append(file.stem)
```

### 2. Run Whole-Spectrum PCA

```python
from xas_ml_modules.xas_spectrum_pca import XASSpectrumPCA

# Initialize analyzer
analyzer = XASSpectrumPCA(
    normalization='standard',
    energy_range=(6912, 7492),  # Adjust for your edge
    n_grid_points=300
)

# Analyze
pca_result = analyzer.analyze_datasets(
    datasets,
    sample_names=sample_names,
    mu_variable='mu_trans'  # or 'mu_ref', 'fluor_corrected'
)

# Check results
print(f"Variance explained: {pca_result.variance_ratio * 100}")
```

### 3. Interpret PCA Components

```python
from xas_ml_modules.xas_experiment_planner import XASExperimentPlanner

# Initialize planner
planner = XASExperimentPlanner(
    edge_energy=7112.0,  # Fe K-edge (adjust for your element)
    xanes_range=(7100, 7160),
    exafs_range=(7160, 7500)
)

# Interpret components
interpretations = planner.interpret_components(pca_result)

print("\nPCA Interpretation:")
for interp in interpretations:
    print(f"  {interp}")
```

### 4. Suggest Next Experiments

```python
# Define your experimental parameters
experimental_params = {
    'pH': [2.2, 5.1, 5.2, 5.3],
    'temperature': [25, 25, 25, 25],
    'ligand': ['malic_acid'] * 4
}

# Try different strategies
strategies = ['boundary', 'maxdist', 'hull']

for strategy in strategies:
    print(f"\n{strategy.upper()} Strategy:")
    
    suggestions = planner.suggest_experiments(
        pca_result,
        experimental_params=experimental_params,
        strategy=strategy,
        n_suggestions=3
    )
    
    for i, sug in enumerate(suggestions):
        print(f"  Suggestion {i+1}:")
        print(f"    PC1: {sug.predicted_scores[0]:.2f}")
        print(f"    Priority: {sug.priority:.2f}")
        print(f"    Reason: {sug.reason}")
```

### 5. Visualize

```python
# Plot experiment planning
planner.plot_experiment_planning(
    pca_result,
    experimental_params=experimental_params,
    suggestions=suggestions,
    pc_x=1,
    pc_y=2,
    color_by='pH',
    save_path='experiment_plan.png'
)
```

---

## Real Example (Copy-Paste Ready)

Complete working example with Fe XAS data:

```python
from pathlib import Path
from xas_reader.aps_xas_reader import load_aps_xas
from xas_ml_modules.xas_spectrum_pca import XASSpectrumPCA
from xas_ml_modules.xas_experiment_planner import XASExperimentPlanner

# Load data
data_dir = Path(r"N:\zhenzhen\C-Steel\Data\XAS data\021526-FeCL2-FeSO4-MA-TA-pH2-5\021526-FeCL2-FeSO4-MA-TA-pH2-5")
datasets = []
sample_names = []

for file in sorted(data_dir.glob("FeCl2-Malic_acid*")):
    if file.suffix != '.hdf':  # ASCII only
        datasets.append(load_aps_xas(file))
        sample_names.append(file.stem)

print(f"Loaded {len(datasets)} spectra")

# Run PCA
analyzer = XASSpectrumPCA()
pca_result = analyzer.analyze_datasets(datasets, sample_names=sample_names)

print(f"\nPCA captured {pca_result.variance_ratio[0]*100:.1f}% variance in PC1")

# Plan experiments
planner = XASExperimentPlanner(edge_energy=7112.0)  # Fe K-edge

# Interpret
interpretations = planner.interpret_components(pca_result, n_components=2)
print("\nPC Interpretation:")
for interp in interpretations:
    print(f"  PC{interp.pc_number}: {interp.interpretation}")

# Extract pH from sample names
ph_values = []
for name in sample_names:
    if 'pH' in name:
        idx = name.find('pH')
        ph_str = ''.join(c for c in name[idx+2:] if c.isdigit() or c == '.')
        ph_values.append(float(ph_str) if ph_str else 0.0)

experimental_params = {'pH': ph_values}

# Suggest next experiments
suggestions = planner.suggest_experiments(
    pca_result,
    experimental_params=experimental_params,
    strategy='boundary',  # Find transition pH
    n_suggestions=5
)

print("\nTop 5 Experiment Suggestions:")
for i, sug in enumerate(suggestions[:5]):
    print(f"\n{i+1}. Priority: {sug.priority:.2f}")
    print(f"   Target PC1: {sug.predicted_scores[0]:.2f}")
    print(f"   {sug.reason}")
    
    # Estimate pH from PC1 (linear interpolation)
    pc1_range = pca_result.scores[:, 0].max() - pca_result.scores[:, 0].min()
    ph_range = max(ph_values) - min(ph_values)
    estimated_ph = (
        min(ph_values) +
        (sug.predicted_scores[0] - pca_result.scores[:, 0].min()) / pc1_range * ph_range
    )
    print(f"   → Suggested pH: {estimated_ph:.1f}")

# Visualize
planner.plot_experiment_planning(
    pca_result,
    experimental_params=experimental_params,
    suggestions=suggestions,
    color_by='pH',
    save_path='fe_experiment_plan.png'
)

print("\n✓ Plot saved to fe_experiment_plan.png")
```

---

## Strategy Selection Guide

**Choose your strategy based on your scientific goal:**

| Goal | Best Strategy | Why |
|------|--------------|-----|
| Find transition pH/potential | `'boundary'` | Samples between clusters |
| Explore new chemistry | `'maxdist'` | Farthest from existing data |
| Systematic coverage | `'hull'` | Expands explored region |
| Fill time series gaps | `'trajectory'` | Requires time parameter |

**Pro tip:** Run all strategies and rank by priority!

---

## Interpreting the Results

### PC Loadings → Chemistry

After running PCA, look at the loadings plot:

```python
# Plot loadings
analyzer.plot_loadings(pca_result, save_path='loadings.png')
```

**What to look for:**

| Loading Pattern | Spectral Region | Physical Meaning |
|----------------|-----------------|------------------|
| Peak at ~7112 eV | Edge | Oxidation state (Fe²⁺/Fe³⁺) |
| Peak at ~7130 eV | White line | Coordination number |
| Oscillations > 7160 eV | EXAFS | Bond distances |
| Pre-edge peak ~7110 eV | Pre-edge | Symmetry/geometry |

### PC Scores → Sample Grouping

Look at the scores:

```python
# Plot scores
analyzer.plot_scores(pca_result, save_path='scores.png')
```

**What you'll see:**

- **Clusters**: Different chemical states
- **Trajectories**: Reaction progress or systematic changes
- **Outliers**: Unusual samples (check for errors or new chemistry!)

### Suggested Experiments → Action

Each suggestion has:
- **PC scores**: Where in PCA space
- **Distance**: How far from existing data (higher = more novel)
- **Priority**: Recommendation strength
- **Reason**: Why this experiment is suggested

**Best practice:** Look at top 3-5 suggestions, choose the most experimentally feasible one.

---

## Common Patterns

### Pattern 1: Two Clusters

**Scenario:** pH 2 samples cluster together, pH 5 samples cluster together

**PCA Space:**
```
  PC2
   |
   |  ○○○  (pH 5)
   |
   |
---+-------------PC1
   |
   |
   |         ○○○  (pH 2)
```

**What to do:**
- Use `'boundary'` strategy
- Suggests experiments between clusters (pH 3-4)
- **Goal:** Find transition point

### Pattern 2: Trajectory

**Scenario:** Reaction time series

**PCA Space:**
```
  PC2
   |
   |      ○ (t=100)
   |     /
   |    ○ (t=10)
---+---○-----------PC1
   |  (t=0)
```

**What to do:**
- Use `'trajectory'` strategy
- Suggests experiments at intermediate times
- **Goal:** Resolve reaction mechanism

### Pattern 3: Scattered Points

**Scenario:** Multi-parameter space, sparse sampling

**PCA Space:**
```
  PC2
   |
   | ○     ○
   |
   |   ○       ○
---+-------------PC1
   |
   |  ○   ○
```

**What to do:**
- Use `'hull'` or `'maxdist'` strategy
- Systematically fill gaps
- **Goal:** Complete parameter space exploration

---

## Tips & Tricks

### 1. Start Small

Test with 4-6 samples first:
- Verify PCA captures known chemistry
- Check if PC1 explains > 80% variance
- Validate interpretation before scaling up

### 2. Check Variance

```python
cumsum = np.cumsum(pca_result.variance_ratio)
print(f"95% variance in {np.where(cumsum > 0.95)[0][0] + 1} PCs")
```

- < 3 PCs needed → Simple system
- > 5 PCs needed → Complex, multi-parameter system

### 3. Name Samples Informatively

Good: `FeCl2_Malic_acid_pH2.2_25C`  
Bad: `Sample001`

Why? Easier to extract experimental parameters automatically.

### 4. Iterate

After each batch of experiments:
1. Add new spectra
2. Re-run PCA (more data → better PCA)
3. Update suggestions (refined understanding)
4. Plan next batch

### 5. Combine with Domain Knowledge

PCA finds patterns. You know the chemistry.

**Example:**
- PCA suggests pH 8.5
- You know: Sample precipitates at pH > 8
- **Action:** Test pH 7.5 instead (closest feasible condition)

---

## Troubleshooting

### Problem: "PC1 explains 100% variance"

**Cause:** Limited spectral diversity (very similar samples)

**Solution:**
- This is actually good! Strong trend detected.
- Use `'boundary'` if you see separation
- Or add more diverse samples

### Problem: "All suggestions have low priority"

**Cause:** You've already explored most of the accessible space

**Solution:**
- Consider new experimental parameters (temperature, etc.)
- Or focus on refinement (boundary sampling)

### Problem: "Suggestions are chemically unrealistic"

**Cause:** PCA doesn't know experimental constraints

**Solution:**
- Filter suggestions:
```python
def is_feasible(sug):
    estimated_ph = estimate_ph(sug)
    return 1 < estimated_ph < 12

feasible = [s for s in suggestions if is_feasible(s)]
```

### Problem: "Can't install scipy"

**Cause:** Convex hull requires scipy

**Solution:**
- Install: `pip install scipy`
- Or use `'maxdist'` or `'boundary'` strategies (don't need scipy)

---

## Output Files

The planner creates:

```
experiment_planning/
├── planning_maxdist.png      # MaxDist strategy visualization
├── planning_boundary.png     # Boundary strategy visualization
├── planning_hull.png         # Hull strategy visualization
└── planning_trajectory.png   # Trajectory strategy (if applicable)
```

Each plot shows:
- **Blue shaded region**: Explored PCA space (convex hull)
- **Circles**: Existing experiments (color-coded by parameter)
- **Yellow stars**: Suggested experiments (size = priority)
- **Labels**: Sample names and suggestion numbers

---

## Next Steps

1. **Run the example above** with your data
2. **Check PC interpretations** - Do they make chemical sense?
3. **Review top 3-5 suggestions** for each strategy
4. **Select 1-2 experiments** that are feasible and interesting
5. **Run them** and add to your dataset
6. **Re-run PCA** with expanded data
7. **Repeat!**

---

## Full Workflow Script

Save as `plan_experiments.py`:

```python
#!/usr/bin/env python
"""Plan next XAS experiments using PCA."""

from pathlib import Path
from xas_reader.aps_xas_reader import load_aps_xas
from xas_ml_modules.xas_spectrum_pca import XASSpectrumPCA
from xas_ml_modules.xas_experiment_planner import XASExperimentPlanner

def main(data_dir, edge_energy=7112.0):
    # Load data
    print("Loading data...")
    datasets = []
    names = []
    
    for file in sorted(Path(data_dir).glob("*")):
        if file.suffix != '.hdf':  # ASCII only
            try:
                datasets.append(load_aps_xas(file))
                names.append(file.stem)
            except:
                pass
    
    print(f"  Loaded {len(datasets)} spectra")
    
    # PCA
    print("\nRunning PCA...")
    analyzer = XASSpectrumPCA()
    result = analyzer.analyze_datasets(datasets, sample_names=names)
    print(f"  PC1: {result.variance_ratio[0]*100:.1f}% variance")
    
    # Plan
    print("\nPlanning experiments...")
    planner = XASExperimentPlanner(edge_energy=edge_energy)
    
    # Interpret
    interp = planner.interpret_components(result)
    print("\nPC Interpretation:")
    for i in interp:
        print(f"  {i}")
    
    # Suggest
    print("\nTop Suggestions:")
    for strategy in ['boundary', 'maxdist']:
        sug = planner.suggest_experiments(result, strategy=strategy, n_suggestions=1)[0]
        print(f"  [{strategy}] PC1={sug.predicted_scores[0]:.2f}, priority={sug.priority:.2f}")
    
    # Plot
    planner.plot_experiment_planning(result, save_path='plan.png')
    print("\n✓ Plot saved to plan.png")

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    main(data_dir)
```

**Usage:**
```bash
python plan_experiments.py your/data/directory
```

---

## Resources

- **Full Guide**: `EXPERIMENT_PLANNING_GUIDE.md`
- **PCA Guide**: `WHOLE_SPECTRUM_PCA_GUIDE.md`
- **Reader Docs**: `APS_XAS_READER_DOCUMENTATION.md`
- **Test Script**: `test_experiment_planning.py`

---

**Ready to start? Copy the "Real Example" code above and run it on your data!**
