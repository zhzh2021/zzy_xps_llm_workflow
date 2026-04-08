# XAS ML Analysis - Quick Start

## Main Entry Point

**File**: `xas_ml_modules/xas_ml_standalone_main.py`

This is the **primary interface** for all ML analysis. Use this file for:
- Feature-based ML (PCA, clustering, correlations)
- Whole-spectrum PCA analysis
- Experiment planning and suggestions
- Automated visualization generation

## Usage

### Simplest Call (Run Everything)

```bash
python xas_ml_modules/xas_ml_standalone_main.py
```

### Common Options

```bash
# Feature-based ML only (faster)
python xas_ml_modules/xas_ml_standalone_main.py --mode features

# Spectrum PCA + planning only
python xas_ml_modules/xas_ml_standalone_main.py --mode spectrum

# See all options
python xas_ml_modules/xas_ml_standalone_main.py --help
```

## What You Get

### Outputs in `04_ml_analysis/`

**Data Files** (`analysis_results/`):
- Feature matrix, metadata, PCA results
- Cluster assignments
- Feature-metadata correlations

**Spectrum PCA** (`spectrum_pca_results/`):
- PCA scores and loadings (500 energy points)
- Component interpretations
- Experiment suggestions (ranked by priority)

**Visualizations** (`analysis_plots/`):
- PCA scree, scores, loadings plots
- **Conditions overlay** (multi-panel showing all experimental variables)
- Experiment planning plot (with suggestions)

## For AI Agents

### Python Call

```python
import subprocess

result = subprocess.run([
    'python', 
    'xas_ml_modules/xas_ml_standalone_main.py'
], capture_output=True, text=True)

# Check success
if '[OK]' in result.stdout:
    print("Analysis complete!")
```

### Terminal Call

```bash
cd xas_ml_modules
python xas_ml_standalone_main.py --mode all
```

## Requirements

**Input Data**:
- Feature extraction completed: `03_feature_extraction/*_features.json`
- Normalized spectra: `02_analyzed_data/normalized_data/*_analyzed.csv`

**Minimum**: 2 samples for any analysis

## Expected Results (42 Fe K-edge Samples)

- **Feature PCA**: 95% variance in 5 components
- **Spectrum PCA**: 99% variance in 5 components  
- **Clustering**: 4 clusters, silhouette = 0.56
- **Runtime**: ~10 seconds total

## Documentation

Full details: [ML_MAIN_ENTRY_GUIDE.md](ML_MAIN_ENTRY_GUIDE.md)

## Quick Troubleshooting

**"Loaded 0 samples"**: Check that feature extraction ran successfully

**"Loaded 0 spectra"**: Check that data analysis stage completed

**Both failing**: Check project_root path in script output

---

**For detailed usage, examples, and API reference, see ML_MAIN_ENTRY_GUIDE.md**
