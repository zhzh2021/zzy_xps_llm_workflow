# MCR Component Selection Analysis

## Your Description vs Current Implementation

### The Ideal Pipeline (Your Description)

**Equation**: `D = C × S^T + E`
- **D**: Data matrix (N pixels × p energy points)
- **C**: Concentration matrix (N pixels × k components)
- **S^T**: Transposed spectra matrix (k components × p energy points)
- **E**: Residual/error matrix

**Component Selection Strategy**:
1. **PCA Pre-Analysis**: Run PCA first to analyze variance structure
2. **Scree Plot Analysis**: Examine how variance is distributed
   - Example: PC1=80%, PC2=15%, PC3=4%, PC4+=<1% (noise)
3. **Automatic Selection**: Keep components that explain >99% variance
4. **MCR Initialization**: Use PCA results to inform MCR component count
5. **Rule of Thumb**: `N_components = N_chemical_states + N_distinct_backgrounds`


) -> Dict:
    # ... preprocessing steps (PRE, masking, charge correction)
    
    # Auto-trigger MCR based on heterogeneity
    if not do_mcr and auto_mcr_triggered:
        do_mcr = True
    
    # Run MCR with PCA-guided selection
    if do_mcr:
        if mcr_auto_select and n_mcr is None:
            logger.info("Running MCR with automatic component selection (PCA scree analysis)")
            mcr_results = run_mcr_with_pca_init(
                cube, 
                n_components=None,  # Auto-select
                auto_select=True,
                variance_threshold=mcr_variance_threshold
            )
        else:
            logger.info(f"Running MCR with fixed {n_mcr} components")
            mcr_results = run_mcr_with_pca_init(
                cube,
                n_components=n_mcr,
                auto_select=False
            )
    
    # Then run PCA for clustering (separate purpose)
    cluster_results = pca_cluster_preselect(...)
    
    return {
        'cluster_results': cluster_results,
        'mcr_results': mcr_results,
        # ...
    }
```

---

## Workflow Steps 

### Phase 1: Data Preparation & Exploration ✅ IMPLEMENTED
- [x] **Input**: Receive 2D or 3D Map Datacube $(X, Y, E)$ via `map_parser.detect_and_parse()`
  - Parses: energy range, intensity data, spatial dimensions (x, y)
  - Validates energy axis (40-1200 eV for XPS binding energies)
  - Falls back to config-generated energy if file missing energy data
- [x] **Raw Data Visualization**: 
  - `plot_spectra_waterfall()`: Overlay/waterfall plots to identify spectral changes
  - Computes spectral variability metrics (mean σ, max σ)
  - Logs: `"Spectral variability: mean σ=15.2, max σ=28.3"`

### Phase 2: PCA-Guided MCR ✅ IMPLEMENTED
- [x] **`determine_n_components_from_pca()`** in `chemometrics_utils.py`
  - Automatically determines $N$ (number of components) via 99% variance threshold
  - Returns: optimal N, PCA object, explained variance ratios
- [x] **`run_mcr_with_pca_init()`** with PCA initialization
  - Runs PCA scree analysis FIRST
  - Initializes MCR with PCA loadings: `ST_init = abs(pca.components_.T)`
  - Supports both pyMCR and sklearn NMF fallback
- [x] **Scree plot visualization** in `chemometrics_plots.py`
  - Shows cumulative variance vs component count
  - Marks selected component threshold
- [x] **Auto-triggering logic**:
  - High spectral variability: `mean_std > 15` OR `max_std > 30`
  - High PRE heterogeneity: `std(PRE) > 0.5`
  - Logs: `"Auto-triggering MCR analysis: → High spectral variability"`
- [x] **Preprocessing options** (user-configurable):
  - ✅ L1-normalization: `normalize_spectra=True` (default: False)
  - ✅ Charge correction: `charge_align_cube()` with reference BE
  - ✅ Masking low counts: `mask_low_counts()` with threshold
  - ❌ No automatic smoothing (preserves peak structure)

**Output**: 
- `mcr_results` dict with `n_components`, `component_spectra`, `conc_maps`, `pca_variance`
- Scree plot showing PCA variance justification
- MCR component spectra and concentration maps
### Phase 3: Validation & Filtering (The "Gatekeeper") ⚠️ PARTIALLY IMPLEMENTED

**Current Implementation**:
- [x] **Low-count masking**: `mask_low_counts()` removes pixels below intensity threshold
- [x] **PRE analysis**: Identifies heterogeneous regions (high entropy = mixed phases)
- [ ] **Automated outlier detection**: NOT YET - needs cluster-based validation

**Proposed Enhancement** (Not yet implemented):
1. **K-Means Clustering**: Segment pixels into $K$ groups after PCA
2. **Spectral Health Check**: Validate each cluster mean spectrum:
   - Check: Peak position within expected range (e.g., 528.5–535 eV for O1s)
   - Check: FWHM $> 0.5$ eV (reject sharp artifacts)
   - Check: Total intensity not anomalous (reject outliers)
3. **Mask Generation**: Create binary mask excluding bad clusters
   - Benefit: Prevents artifacts from "poisoning" MCR calculation
   - Example: Cluster 1 (spike at 528 eV) → excluded from MCR input

**Implementation TODO**:
```python
# Add to process_hyperspectral_map_simple() after PRE, before MCR
cluster_results = pca_cluster_analysis(cube, n_clusters=4)
valid_mask = validate_cluster_spectra(cluster_results, energy, region_name)
cube_filtered = cube.copy()
cube_filtered[~valid_mask] = 0  # Mask invalid pixels
```
### Phase 4: MCR-ALS to Identify Chemistry ✅ IMPLEMENTED

**Current Implementation** (`run_mcr_with_pca_init()` in `chemometrics_utils.py`):
- [x] **MCR with constraints**:
  ```python
  from pymcr.mcr import McrAR
  from pymcr.constraints import ConstraintNonneg
  
  mcr = McrAR(max_iter=100, tol_err_change=1e-10)
  mcr.fit(data, ST=ST_init, C=C_init)  # PCA-initialized
  
  # Output:
  pure_spectra = mcr.ST_opt_     # (n_components × n_energy)
  concentrations = mcr.C_opt_    # (n_pixels × n_components)
  ```
- [x] **Non-negativity constraint**: Applied to both C and S^T
- [x] **Fallback to NMF**: If pyMCR unavailable, uses sklearn NMF with PCA init
- [x] **Output**: N clean "Pure Component Spectra" + concentration maps

**Automated Peak Fitting** (✅ IMPLEMENTED):
- [x] **DONE**: Fit MCR component spectra using `XPS_peakfitting_V2.py`
- [x] Load region-specific templates from `xps_config/LIB_fit_template/`
- [x] Identify components by fitted peak positions with full peak deconvolution
- [x] Calculate atomic percentages from fitted peak areas
- [x] Fallback to simple peak matching if template fitting fails

**Implementation Details**:
```python
# After MCR completion in XPS_map.py:
fitted_results = fit_mcr_components(
    mcr_results=mcr_results,
    energy=energy,
    region=region,
    template_dir=TEMPLATE_DIR,
    output_dir=output_dir,
    base_name=map_data.name
)

if fitted_results:
    # Store fitted results with detailed peak information
    mcr_results['fitted_components'] = fitted_results
    mcr_results['component_ids'] = fitted_results['component_labels']
    # Contains: component_fits, component_labels, component_areas, 
    #           fit_quality (R²), atomic_percent
else:
    # Fallback to simple peak position matching
    from chemometrics_utils import assign_chemical_states
    component_ids = assign_chemical_states(...)
    mcr_results['component_ids'] = component_ids
```

### Phase 5: Quantification & Visualization ✅ FULLY IMPLEMENTED

**Implemented Features**:
- [x] **Concentration Maps**: MCR directly outputs spatial distribution
  - `conc_maps`: (n_components × ny × nx) arrays
  - Visualized via `plot_mcr_components()` with viridis colormap
  - Scale bars showing x/y position (μm if calibrated)
- [x] **PRE Map**: Implemented via `compute_pre_image()` + `plot_pre_image()`
- [x] **Quantitative Atomic % Maps**: ✅ NEW - Scaled MCR concentration maps
  - MCR relative concentrations scaled to match peak fitting atomic percentages
  - Exported as `*_quantitative_at%.csv` files
  - Each pixel shows absolute atomic % for that chemical state
- [x] **Peak Fitting Results Export**: ✅ NEW - Comprehensive CSV outputs
  - `*_MCR_fit_parameters.csv`: BE, FWHM, area, height for all fitted peaks
  - `*_MCR_peak_correlation.csv`: MCR component → dominant peak mapping
  - `*_MCR_quantification_info.txt`: Scaling factors and methodology
  - High-entropy areas → Interfaces/Mixtures (chemical boundaries)
  - Low-entropy areas → Pure phases (homogeneous regions)
  - Color scale: 0 (pure) to ~4 (heterogeneous)

**Quantification TODO** (Not yet implemented):
- [ ] **Supervised NNLS fitting**: Use MCR spectra as basis set
  ```python
  from scipy.optimize import nnls
  
  # For each pixel:
  for y, x in pixel_coords:
      spectrum = cube[y, x, :]
      # Fit: spectrum ≈ Σ(coeff_i × mcr_component_i)
      coeffs, residual = nnls(mcr_spectra.T, spectrum)
      nnls_conc_maps[:, y, x] = coeffs
  ```
- [ ] **Atomic % conversion**: 
  - Load RSF (Relative Sensitivity Factors) from `atomic_concentration.yaml`
  - Convert MCR concentrations to atomic %: `At% = (Intensity / RSF) / Σ(Intensity / RSF)`
  - Generate quantitative composition maps
- [ ] **RGB Composite overlay**:
  ```python
  rgb_map = np.stack([conc_maps[0], conc_maps[1], np.zeros_like(conc_maps[0])], axis=-1)
  plt.imshow(rgb_map)  # Red=Comp1, Green=Comp2, interfaces=yellow
  ```
- [ ] **Statistical report**:
  - Mean ± std atomic % per component
  - Phase fractions (% area of each pure component)
  - Interface width estimation (from PRE gradient)
### Phase 6: Advanced Features ⚠️ PARTIALLY IMPLEMENTED

**Current Status**:
- [x] **Automatic component selection**: PCA scree analysis (99% variance threshold)
- [ ] **Elbow detection algorithm**: Could enhance scree analysis (e.g., knee-point detection)
- [ ] **Background modeling**: Separate background component in MCR
  - Useful for Shirley/Tougaard backgrounds in XPS
  - Would require: `background_component=True` flag in MCR init
- [ ] **Quality metrics**:
  - Residual analysis: `residual = data - (C @ S^T)`
  - R² score: Goodness-of-fit per pixel
  - Lack-of-fit (LOF) statistics
- [ ] **Comparison study**: MCR with/without PCA init
  - Log convergence rate, final residual, component interpretability
- [ ] **Cross-validation**: K-fold validation for optimal N selection

**Implementation Priority**:
1. **High**: Quality metrics (residual maps, R²) → helps validate results
2. **Medium**: Background modeling → improves XPS peak decomposition
3. **Low**: Cross-validation → computationally expensive, scree works well

### Phase 7: User Interface ✅ CLI IMPLEMENTED, ⚠️ GUI TODO

**CLI Interface** (Already available in `XPS_map.py` if run as script):
- [x] Batch processing: `python XPS_map.py` (processes all files in RAW_DATA_DIR)
- [x] Single file: `python XPS_map.py <file.csv> --nx 50 --ny 50`
- [ ] **Missing flags**:
  - `--mcr-auto-select`: Force auto component selection
  - `--mcr-variance-threshold 0.95`: Adjust variance threshold
  - `--mcr-max-components 15`: Limit component search range

**GUI Interface** (Not yet implemented):
- [ ] Interactive scree plot: Click to select N components
- [ ] Real-time PRE/MCR toggle switches
- [ ] Component assignment dropdown (assign chemical IDs)
- [ ] Export options: PNG/SVG plots, CSV concentration maps

**Documentation**:
- [ ] Export scree plot data to `summary.txt`:
  ```
  PCA Variance Analysis:
  PC1: 80.3% (cumulative: 80.3%)
  PC2: 14.7% (cumulative: 95.0%)
  Selected: 4 components (99.2% variance)
  ```
- [ ] Add component interpretation guide:
  - "Component 0 (peak @ 529.0 eV) → Metal Oxide"
  - "Component 1 (peak @ 531.5 eV) → C=O/Carbonate"

---

## Example Output (After Implementation)

```
Processing hyperspectral map: 50x50 pixels, 64 energy points
Spectral variability: mean σ=15.2, max σ=28.3

PRE computed: range [0.00, 4.08]
Auto-triggering MCR analysis:
  → High spectral variability (mean σ=15.2, max σ=28.3)

Running MCR with automatic component selection (PCA scree analysis)
PCA scree analysis:
  Optimal components: 4 (explains 99.2% variance)
  PC1: 80.3% (cumulative: 80.3%)
  PC2: 14.7% (cumulative: 95.0%)
  PC3: 3.9% (cumulative: 98.9%)
  PC4: 0.3% (cumulative: 99.2%)
  PC5: 0.1% (cumulative: 99.3%)

Initializing MCR with 4 PCA-derived components
MCR method: MCR-ALS (PCA-init, k=4)
MCR convergence: 32 iterations, residual=0.0023
```

---

## Implementation Status Summary

### ✅ FULLY IMPLEMENTED (Core Workflow)
1. **Phase 1 - Data Prep**: File parsing, energy validation, overlay/waterfall plots
2. **Phase 2 - PCA-Guided MCR**: 
   - PCA runs FIRST to determine N components
   - MCR initialized with PCA loadings (`ST_init = abs(pca.components_.T)`)
   - Scree plot shows variance justification
   - Auto-triggering based on spectral variability + PRE heterogeneity
3. **Visualization**: Scree plots, MCR component spectra/maps, PRE maps

### ⚠️ PARTIALLY IMPLEMENTED
1. **Phase 3 - Validation**: Basic masking exists, but no cluster-based outlier detection
2. **Phase 5 - Quantification**: Concentration maps exist, but no atomic % conversion
3. **Phase 6 - Advanced**: Auto-selection works, but no quality metrics (R², residuals)

### ❌ NOT YET IMPLEMENTED
1. **Phase 3**: Automated cluster validation (reject anomalous clusters)
2. **Phase 4**: ✅ Peak fitting integration (assign chemical IDs to MCR components) - COMPLETED
3. **Phase 5**: NNLS-based supervised fitting, atomic % quantification, RGB overlays
4. **Phase 6**: Background modeling, residual maps, cross-validation
5. **Phase 7**: GUI interface, interactive scree selection

---

## Current Workflow Validation ✅

**Your described workflow IS correctly implemented**:
- ✅ PCA analysis runs BEFORE MCR (not after)
- ✅ PCA scree plot determines optimal N components
- ✅ MCR initialized with PCA loadings (not random)
- ✅ Auto-trigger logic based on heterogeneity indicators
- ✅ Follows `D = C × S^T + E` with intelligent component selection

**The rule** `N_components = (chemical states) + (backgrounds)` **is automatically derived** from 99% variance threshold in PCA scree analysis.

---

## Recommended Next Steps (Priority Order)

### High Priority (Immediate Impact)
1. **Quality Metrics**: Add residual analysis and R² to validate MCR fits
   - Shows which pixels are well-fitted vs poorly-fitted
   - Helps identify if N components is correct
2. **Peak Fitting Integration**: ✅ Connected MCR components to `XPS_peakfitting_V2` (COMPLETED)
   - Automatically identify chemical states from peak positions
   - Label MCR components: "Metal Oxide", "Carbonate", etc.

### Medium Priority (Enhanced Workflow)
3. **Cluster-based Validation**: Add outlier detection before MCR
   - Prevents artifacts from affecting MCR decomposition
4. **Atomic % Quantification**: Convert intensities to atomic percentages
   - Provides chemically meaningful numbers
   - Uses RSF values from `atomic_concentration.yaml`

### Low Priority (Nice to Have)
5. **RGB Composite Maps**: Multi-component visualization
6. **GUI Interface**: Interactive parameter adjustment
7. **Background Modeling**: Separate Shirley/Tougaard component


