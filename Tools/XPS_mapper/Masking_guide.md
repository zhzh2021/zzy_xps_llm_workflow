# Spatial Masking Guide for XPS Map Analysis

## Overview

Spatial masking removes unwanted pixels from XPS hyperspectral maps before chemometric analysis (PCA, MCR, clustering). This improves data quality by excluding noise, artifacts, dead pixels, and background regions that would otherwise contaminate results.

**Why masking matters:**
- MCR convergence failures often stem from including too much noise
- Low-quality pixels dilute variance explained by meaningful components  
- Background substrate signals can dominate and obscure sample chemistry

---

## Decision Tree: Which Masking Method?

### 1. Intensity Masking (ALWAYS START HERE)

**What:** Removes dead pixels or empty areas with counts below threshold

**When:** Every analysis - standard first step to define sample area

**Parameter:** `intensity_mask_threshold: 10.0`

**Status:** ✅ Implemented

---

### 2. PCA-Score Masking (FIRST CHOICE FOR NOISY DATA)

**What:** Uses PC1 score map to isolate pixels correlating with spectral peak shape

**When to use:**
- MCR fails to converge (0 iterations)  
- PC1 variance explained < 20%
- High noise-to-signal ratio visible in maps

**Why it works:** PCA extracts peak direction from variance. Masking low PC1 scores removes pixels lacking coherent spectral features.

**Parameter:** `pca_mask_threshold: 0.5`

**Status:** ⚠️ Not yet implemented

---

### 3. Cluster Masking (ADVANCED - USE WITH CAUTION)

**What:** Isolates pixels belonging to specific chemical states from K-means clustering

**When to use:**
- Silhouette score > 0.3 (confirms clusters are meaningful)
- Clear spatial separation of chemical phases  
- After noise removal via intensity/PCA masking

**When NOT to use:**
- Silhouette score < 0.2 (clusters are arbitrary, not chemical)
- Small datasets (<100 pixels)
- Uncertain which clusters are signal vs. noise

**Risk:** If clusters aren't chemically meaningful, masking produces "garbage in, garbage out" for MCR

**Parameter:** `focus_clusters: [2]`

**Status:** ✅ Implemented

---

## Recommended Workflow

1. **Baseline:** Run with only intensity masking
2. **Diagnose:** Check MCR convergence, PC1 variance, Silhouette score  
3. **Intervene:**
   - MCR fails → try PCA-score masking (when available)
   - Silhouette >0.3 → try cluster masking
4. **Compare:** Validate that masking improves results

---

## Cluster Masking Implementation

### Method 1: Config File (Recommended)

Add `focus_clusters` to element-specific region:

```yaml
regions:
  F1s:
    peak_centers: [687.0]
    peak_widths: [2.5]
    analysis_range: [682.0, 690.0]
    intensity_mask_threshold: 10.0
    focus_clusters: [2]  # After initial clustering showed cluster 2 = signal
```

Or globally:

```yaml
global_processing:
  focus_clusters: [2, 3]
```

**Priority:** Element-specific > Global > Function parameter

### Method 2: Direct Parameter

```python
results = process_hyperspectral_map_simple(
    map_data=map_data,
    output_dir=output_dir,
    config=config,
    n_clusters=4,
    cluster_mask=[2],  # Keep only cluster 2
    do_mcr=True
)
```

---

## How It Works

1. **Initial Clustering:** K-means identifies spatial patterns  
2. **Cluster Selection:** User specifies IDs to keep (0-indexed)
3. **Mask Creation:** Boolean mask (True = selected clusters)
4. **Spatial Zeroing:** Pixels NOT in selected clusters → 0
5. **Downstream Analysis:** PCA/MCR/NMF process only non-zero pixels

**Outputs:**
- `{name}_cluster_spatial_mask.png`: Binary mask (white=kept, black=excluded)
- Log: `Cluster mask: keeping 23761 pixels, masking 41775 pixels`

---

## Step-by-Step Workflow

### Step 1: Initial Analysis (No Cluster Mask)

```python
results = process_hyperspectral_map_simple(
    map_data=map_data,
    output_dir=output_dir / "initial",
    n_clusters=4,
    cluster_mask=None
)
```

**Check:**
- Cluster spectra: Identify signal vs. noise
- Silhouette score: >0.3 = meaningful clusters  
- Cluster maps: Spatial distribution

---

### Step 2: Identify Signal Clusters

Classify each cluster:

| Cluster | Characteristics | Type | Action |
|---------|----------------|------|--------|
| 0 | Low intensity, flat | Noise | Exclude |
| 1 | Random spikes | Artifacts | Exclude |
| 2 | **Clean peak, high signal** | **Main** | **Keep** |
| 3 | Broad, substrate | Background | Exclude |

**Example:**
- Cluster 0: n=11260, noise → Exclude
- Cluster 1: n=14211, spikes → Exclude
- Cluster 2: n=23761, **signal** → **Keep**  
- Cluster 3: n=16304, background → Exclude

**Decision:** `focus_clusters: [2]`

---

### Step 3: Apply Mask

```python
results_masked = process_hyperspectral_map_simple(
    map_data=map_data,
    output_dir=output_dir / "cluster2_only",
    n_clusters=4,
    cluster_mask=[2],
    do_mcr=True,
    validate_clusters=False  # Skip since manually masking
)
```

---

### Step 4: Compare Results

**Metrics:**
- MCR convergence: iteration count, convergence status
- PCA variance: PC1 variance increase?
- Component quality: cleaner spectra?
- Spatial coherence: clearer patterns?

**Expected improvements:**
- Higher PC1 variance (noise removed)
- Better MCR convergence  
- More interpretable component spectra
- Clearer spatial distributions

---

## Best Practices

1. **Baseline first:** Always compare masked vs. unmasked
2. **Check Silhouette:** Only trust clusters with score >0.3
3. **Combine methods:**
   ```yaml
   F1s:
     intensity_mask_threshold: 10.0  # Dead pixels
     focus_clusters: [2]              # Signal cluster
   ```
4. **Multi-cluster:** Keep multiple if needed: `cluster_mask=[2, 3]`
5. **Document:** Record which clusters kept and why
6. **Validate:** Ensure masking improves metrics

---

## Depth Profile Analysis

Cycles treated as spatial pixels:

```python
from case3_depth_to_map_adaptor import convert_depth_profile_for_mapper

map_data = convert_depth_profile_for_mapper(
    file_path="depth_f1s.csv",
    region_name='F1s',
    debug=False
)

results = process_hyperspectral_map_simple(
    map_data=map_data,
    output_dir=output_dir,
    config=config,
    n_clusters=4,
    cluster_mask=[2],  # Exclude noisy cycles
    do_mcr=True
)
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| "Cluster ID X out of range" | Cluster doesn't exist | Use IDs 0 to n_clusters-1 |
| All pixels masked | Cluster has no/few pixels | Check cluster pixel counts |
| No difference | Cluster already dominates | OK - matches data structure |
| Worse results | Wrong cluster selected | Review spectra, keep multiple clusters |
| Silhouette < 0.2 | Clusters not chemical | Use PCA-score masking instead |

---

## When to Use Each Method

### ✅ Intensity Masking
- Every analysis (required baseline)
- Define sample area vs. empty space

### ✅ PCA-Score Masking (when available)
- MCR convergence failures
- High noise overwhelming signal  
- PC1 variance < 20%

### ✅ Cluster Masking  
- Silhouette > 0.3 (meaningful clusters)
- Clear signal/noise spatial separation
- Distinct chemical phases

### ❌ Don't Use Cluster Masking
- Silhouette < 0.2 (arbitrary clusters)
- All clusters meaningful
- Uncertain signal vs. noise
- Small datasets (<100 pixels)
- Spatial heterogeneity IS research question

---

## Configuration Reference

```yaml
regions:
  F1s:
    analysis_range: [682.0, 690.0]
    intensity_mask_threshold: 10.0    # Dead pixel removal
    focus_clusters: [2]                # Cluster isolation
    # pca_mask_threshold: 0.5          # Coming soon

global_processing:
  intensity_mask_threshold: 5.0
  focus_clusters: [2, 3]
```

---

## Examples

### Example 1: F1s Map with Noisy Background

```yaml
# Config
regions:
  F1s:
    analysis_range: [683.0, 690.0]
    intensity_mask_threshold: 10.0
    focus_clusters: [2]
```

```python
# Processing (focus_clusters read from config)
results = process_hyperspectral_map_simple(
    map_data=f1s_map,
    output_dir=Path("output/f1s_masked"),
    config=config,
    n_clusters=4,
    do_mcr=True
)
```

### Example 2: Multi-Phase Li1s

```yaml
# Config: Two chemical states
regions:
  Li1s:
    focus_clusters: [2, 3]  # LiF + Li₂O phases
```

### Example 3: Override Config

```python
# Test different cluster
results_test = process_hyperspectral_map_simple(
    map_data=map_data,
    cluster_mask=[1],  # Override config
    n_clusters=4,
    do_mcr=True
)
```

---

## Related Documentation

- [TWO_STAGE_QUALITY_GUIDE.md](TWO_STAGE_QUALITY_GUIDE.md) - Cluster validation, quality metrics
- [TOUGAARD_IMPLEMENTATION.md](TOUGAARD_IMPLEMENTATION.md) - Background subtraction  
- [USER_GUIDE.md](USER_GUIDE.md) - General XPS workflow
- [PUBLICATION_PLOTS_GUIDE.md](PUBLICATION_PLOTS_GUIDE.md) - Visualization
