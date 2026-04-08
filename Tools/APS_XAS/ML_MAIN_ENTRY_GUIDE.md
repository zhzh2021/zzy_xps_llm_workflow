# XAS ML Analysis - Main Entry Point Guide

## Overview

**File**: `xas_ml_modules/xas_ml_standalone_main.py`

This is the **MAIN ENTRY POINT** for all machine learning analysis in the XAS workflow. It provides a unified, easy-to-use interface for both users and AI agents.

## Quick Start

### Run Complete Analysis (Default)

```bash
python xas_ml_modules/xas_ml_standalone_main.py
```

This runs:
- Feature-based ML (PCA, clustering, correlations)
- Whole-spectrum PCA
- Experiment planning with conditions overlay
- All visualizations

### Run Specific Modes

```bash
# Feature-based ML only
python xas_ml_modules/xas_ml_standalone_main.py --mode features

# Whole-spectrum PCA + planning only
python xas_ml_modules/xas_ml_standalone_main.py --mode spectrum

# Experiment planning only
python xas_ml_modules/xas_ml_standalone_main.py --mode planning
```

## Command-Line Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--mode` | `all`, `features`, `spectrum`, `planning` | `all` | Analysis mode |
| `--output-dir` | Path string | `project_root/xas_results/04_ml_analysis` | Output directory |
| `--n-components` | Integer | `5` | Number of PCA components |
| `--n-clusters` | Integer | `4` | Number of K-means clusters |

### Examples

```bash
# Custom number of components
python xas_ml_standalone_main.py --n-components 10

# Custom output directory
python xas_ml_standalone_main.py --output-dir /custom/path

# Feature-based ML with 6 clusters
python xas_ml_standalone_main.py --mode features --n-clusters 6
```

## What It Does

### Mode: `all` (Complete Analysis)

**Part 1: Feature-Based ML**
1. Loads extracted features from `03_feature_extraction/`
2. Creates feature matrix (42 samples × 15 features)
3. Runs PCA (standardized)
4. Runs K-means clustering
5. Computes feature-metadata correlations
6. Saves results and generates plots

**Part 2: Whole-Spectrum PCA & Planning**
1. Loads normalized spectra from `02_analyzed_data/normalized_data/`
2. Runs whole-spectrum PCA (7100-7200 eV, 500 points)
3. Interprets principal components
4. Extracts experimental conditions
5. Generates experiment suggestions (3 strategies)
6. Creates comprehensive visualizations:
   - Scree plot
   - Scores plot (colored by pH)
   - Loadings plot
   - **Conditions overlay** (multi-panel)
   - Experiment planning plot

### Mode: `features` (Feature-Based Only)

- Runs only Part 1 (faster, for feature analysis)
- Best for: Understanding feature relationships, clustering

### Mode: `spectrum` or `planning` (Spectrum-Based Only)

- Runs only Part 2 (whole-spectrum analysis)
- Best for: Spectral pattern discovery, experiment design

## Input Requirements

### For Feature-Based ML (`features` or `all`)

**Directory**: `project_root/xas_results/03_feature_extraction/`

**Files**: `*_features.json` (one per sample)

**Minimum**: 2 samples

**Example**:
```
03_feature_extraction/
  ├── FeCl2-Malic_acid_pH2_R1_features.json
  ├── FeCl2-Malic_acid_pH5_R1_features.json
  └── ...
```

### For Whole-Spectrum PCA (`spectrum`, `planning`, or `all`)

**Directory**: `project_root/xas_results/02_analyzed_data/normalized_data/`

**Files**: `*_analyzed.csv` (with columns: energy, mu_cleaned, mu_normalized)

**Minimum**: 2 spectra

**Example**:
```
02_analyzed_data/normalized_data/
  ├── FeCl2-Malic_acid_pH2_R1_dat_analyzed.csv
  ├── FeCl2-Malic_acid_pH5_R1_dat_analyzed.csv
  └── ...
```

## Output Structure

```
04_ml_analysis/
├── analysis_results/
│   ├── feature_matrix.csv
│   ├── metadata.csv
│   ├── pca_summary.json
│   ├── pca_scores.csv
│   ├── pca_loadings.csv
│   ├── cluster_summary.json
│   ├── cluster_assignments.csv
│   └── feature_metadata_correlations.json
│
├── spectrum_pca_results/
│   ├── spectrum_pca_scores.csv
│   ├── spectrum_pca_loadings.csv
│   ├── component_interpretations.json
│   ├── experiment_suggestions.json
│   └── experimental_parameters.csv
│
└── analysis_plots/
    ├── spectrum_pca_scree.png
    ├── spectrum_pca_scores.png
    ├── spectrum_pca_loadings.png
    ├── conditions_overlay.png        ← NEW!
    └── experiment_planning.png
```

## Typical Results

### Feature-Based ML

**Dataset**: 42 samples × 15 features

**PCA**:
- PC1: 46.7% variance (edge properties)
- PC2: 30.8% variance (white line characteristics)
- PC3: 8.9% variance (XANES features)
- Total: 95.2% variance in 5 components

**Clustering**:
- 4 clusters identified
- Silhouette score: 0.562 (good separation)
- Cluster 0: 31 samples (majority)
- Cluster 3: 9 samples (distinct group)

**Correlations**:
- 9 significant correlations (|r| > 0.5, p < 0.05)
- Strongest: white_line_energy ~ ligand_conc (r = 0.775)

### Whole-Spectrum PCA

**Dataset**: 42 spectra (7100-7200 eV, 500 points)

**PCA**:
- PC1: 82.1% variance (major structural differences)
- PC2: 9.7% variance (secondary variations)
- Total: 99.1% in 5 components

**Experiment Suggestions**:
- 5 high-priority suggestions generated
- Top strategy: Boundary (explores cluster transitions)
- Priority score: 33.56

## For AI Agents

### Simple Call

```python
import subprocess

# Run complete analysis
result = subprocess.run([
    'python', 
    'xas_ml_modules/xas_ml_standalone_main.py',
    '--mode', 'all'
], capture_output=True, text=True)

print(result.stdout)
```

### With Options

```python
# Feature-based only with custom clusters
result = subprocess.run([
    'python', 
    'xas_ml_modules/xas_ml_standalone_main.py',
    '--mode', 'features',
    '--n-clusters', '6',
    '--output-dir', '/custom/output'
], capture_output=True, text=True)
```

### Parsing Output

The script prints clear progress indicators:

```
[1/6] Loading extracted features...
  Loaded 42 samples
[2/6] Creating feature matrix...
  Shape: (42, 15)
...
[OK] Results saved to: .../analysis_results
```

Check for:
- `[SKIP]` messages (insufficient data)
- `[OK]` messages (successful steps)
- `[WARN]` messages (non-critical failures)

### Reading Results

```python
import pandas as pd
import json

# Read PCA results
pca_summary = json.load(open('04_ml_analysis/analysis_results/pca_summary.json'))
variance = pca_summary['variance_explained']

# Read cluster assignments
clusters = pd.read_csv('04_ml_analysis/analysis_results/cluster_assignments.csv')

# Read experiment suggestions
suggestions = json.load(open('04_ml_analysis/spectrum_pca_results/experiment_suggestions.json'))
top_suggestion = suggestions[0]  # Highest priority
```

## Troubleshooting

### Error: "Loaded 0 samples"

**Cause**: Wrong directory structure or missing files

**Fix**: 
```bash
# Check feature extraction directory
ls project_root/xas_results/03_feature_extraction/
# Should see *_features.json files
```

### Error: "Loaded 0 spectra"

**Cause**: Missing normalized data

**Fix**:
```bash
# Check analyzed data directory
ls project_root/xas_results/02_analyzed_data/normalized_data/
# Should see *_analyzed.csv files
```

### Warning: "alpha is outside 0-1 range"

**Cause**: matplotlib transparency calculation issue in experiment planning plot

**Fix**: Non-critical - all other plots generated successfully

### Error: "Need at least 2 samples"

**Cause**: Insufficient data for analysis

**Fix**: Run earlier pipeline stages to generate more samples

## Advanced Usage

### Programmatic Access

```python
# Import as module
import sys
sys.path.insert(0, 'path/to/xas_ml_modules')
from xas_ml_standalone_main import (
    run_feature_based_ml,
    run_spectrum_pca_and_planning
)

# Run specific parts
run_feature_based_ml(project_root, output_dir, args)
```

### Custom Experimental Parameters

Modify `extract_experimental_params_from_names()` to parse your naming convention:

```python
def extract_experimental_params_from_names(sample_names):
    # Add custom parsing logic here
    experimental_params = {
        'pH': [],
        'temperature': [],  # New parameter
        'custom_field': []   # New parameter
    }
    # ... parsing logic ...
    return experimental_params
```

## Performance

**Typical Runtime** (42 samples):
- Feature-based ML: ~2 seconds
- Whole-spectrum PCA: ~5 seconds
- Total (mode=all): ~10 seconds

**Memory Usage**: <500 MB

**Output Size**: ~5-10 MB (plots + data)

## Integration

### In GUI

```python
import subprocess

def run_ml_analysis(mode='all'):
    """Run ML analysis from GUI."""
    cmd = ['python', 'xas_ml_modules/xas_ml_standalone_main.py', '--mode', mode]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Stream output to GUI
    for line in process.stdout:
        update_gui_log(line.strip())
    
    returncode = process.wait()
    return returncode == 0
```

### In Pipeline

```bash
#!/bin/bash
# Complete XAS analysis pipeline

# Stage 1: Data reading
python xas_reader/batch_reader.py

# Stage 2: Preprocessing
python xas_analyzer/batch_analyzer.py

# Stage 3: Feature extraction
python xas_feature_extraction/batch_extractor.py

# Stage 4: ML analysis (THIS SCRIPT)
python xas_ml_modules/xas_ml_standalone_main.py --mode all

echo "Pipeline complete!"
```

## Version History

- **v2.0** (2026-03-06): Main entry point with unified interface
  - Command-line arguments
  - Multiple modes (all, features, spectrum, planning)
  - Conditions overlay visualization
  - Comprehensive error handling
  
- **v1.0**: Original standalone feature-based ML script

## See Also

- [CONDITIONS_OVERLAY_GUIDE.md](../CONDITIONS_OVERLAY_GUIDE.md) - Details on multi-panel overlay plots
- [EXPERIMENT_PLANNING_SUMMARY.md](../EXPERIMENT_PLANNING_SUMMARY.md) - Experiment planning theory
- [test_ml_standalone.py](../../test_ml_standalone.py) - Original test script
- [test_spectrum_pca_planning.py](../test_spectrum_pca_planning.py) - Spectrum PCA test script

---

## Quick Reference Card

**Default run**:
```bash
python xas_ml_modules/xas_ml_standalone_main.py
```

**Help**:
```bash
python xas_ml_modules/xas_ml_standalone_main.py --help
```

**Common modes**:
- `--mode all` - Everything (default)
- `--mode features` - Feature ML only
- `--mode spectrum` - Spectrum PCA only

**Outputs**:
- `04_ml_analysis/analysis_results/` - Data files
- `04_ml_analysis/analysis_plots/` - Visualizations
- `04_ml_analysis/spectrum_pca_results/` - Spectrum PCA
