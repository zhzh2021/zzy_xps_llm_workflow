# XAS ML Integration: Implementation Roadmap

**Created:** March 3, 2026  
**Status:** Planning Phase Complete  
**Next Steps:** Begin Phase 1 Implementation

---

## Overview

This roadmap outlines the implementation plan for integrating XASDAML machine learning capabilities into the existing XAS automated processing workflow. The integration adds:

- Feature extraction from normalized XAS spectra
- PCA-based dimensionality reduction for large datasets
- Clustering to identify distinct chemical states
- Trend analysis correlating spectral features with experimental metadata
- Optional simulation validation using FEFF/xraylarch

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│  EXISTING: XAS Preprocessing Pipeline (Per-Spectrum)        │
│  ┌───────┐  ┌──────┐  ┌────────┐  ┌──────────┐  ┌─────┐   │
│  │ Reader│→ │Align │→ │Deglitch│→ │Normalize │→ │  QC │   │
│  └───────┘  └──────┘  └────────┘  └──────────┘  └─────┘   │
│              Implemented in: xas_reader/, xas_analyzer/     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  NEW: ML Analysis Pipeline (Dataset-Level)                  │
│  ┌─────────┐  ┌─────┐  ┌──────────┐  ┌────────┐           │
│  │Features │→ │ PCA │→ │Clustering│→ │ Trends │           │
│  └─────────┘  └─────┘  └──────────┘  └────────┘           │
│              To implement in: xas_ml_modules/               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: Agent-Readable Results + Visualizations            │
│  • JSON report  • CSV exports  • Diagnostic plots           │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation (Week 1-2)

**Goal:** Set up data structures, config system, and basic feature extraction

### Task 1.0: Create Configuration Infrastructure
**File:** `xas_config/xas_ml_settings.yaml`

- [x] Created comprehensive YAML configuration file with all parameters
- [x] Defined sections for: feature_extraction, pca, clustering, trend_analysis, batch_processing, output, logging
- [x] All human-configurable parameters in one place (no hardcoded values)
- [ ] Create ConfigLoader utility class for consistent loading across modules
- [ ] Add config validation schema (optional: use Pydantic)

**Deliverable:** `xas_config/xas_ml_settings.yaml` ✅ created

### Task 1.1: Extend Data Models
**File:** `xas_analyzer/xas_models.py`

- [x] Design complete - Add missing features to `XASFeatures` class:
  - Edge features: `edge_slope`, `pre_edge_area`
  - XANES features: `white_line_fwhm`, `xanes_centroid`
  - Derivative features: `first_derivative_max`, `second_derivative_zero`
  - Statistical features: `spectral_mean`, `spectral_variance`, `spectral_skewness`, `spectral_kurtosis`

- [x] Design complete - Create new models:
  ```python
  class XASDataset(BaseModel): ...
  class PCAAnalysisResult(BaseModel): ...
  class ClusteringResult(BaseModel): ...
  class TrendAnalysisResult(BaseModel): ...
  class XASDatasetAnalysis(BaseModel): ...  # Top-level container
  ```

**Acceptance Criteria:**
- All models have Pydantic validation
- JSON serialization works for numpy arrays (custom encoder)
- Models pass unit tests with synthetic data

---

### Task 1.2: Feature Extractor Module
**New File:** `xas_analyzer/xas_feature_extractor.py`

**Implementation:**
```python
class XASFeatureExtractor:
    """Extract interpretable features from normalized XAS spectra."""
    
    def extract_features(self, 
                        energy: np.ndarray, 
                        mu: np.ndarray,
                        processing_result: Dict) -> XASFeatures:
        """
        Extract all features from normalized spectrum.
        
        Parameters
        ----------
        energy : array
            Energy axis (eV)
        mu : array
            Normalized absorption coefficient
        processing_result : dict
            Contains pre-edge params, e0, edge_step from normalization
            
        Returns
        -------
        XASFeatures
            Complete feature set
        """
        features = {}
        
        # Edge features (from processing_result)
        features['e0'] = processing_result['e0']
        features['edge_step'] = processing_result['edge_step']
        features['edge_slope'] = self._compute_edge_slope(energy, mu)
        features['pre_edge_area'] = self._integrate_pre_edge(energy, mu, features['e0'])
        
        # XANES features
        xanes_features = self._extract_xanes_features(energy, mu, features['e0'])
        features.update(xanes_features)
        
        # Derivative features
        deriv_features = self._extract_derivative_features(energy, mu)
        features.update(deriv_features)
        
        # Statistical features
        stats_features = self._extract_statistical_features(mu)
        features.update(stats_features)
        
        return XASFeatures(**features)
```

**Sub-methods to implement:**
- `_compute_edge_slope()`: Max of first derivative near E0
- `_integrate_pre_edge()`: Area under curve from E0-20 to E0
- `_extract_xanes_features()`: White line detection, XANES centroid
- `_extract_derivative_features()`: 1st and 2nd derivative analysis
- `_extract_statistical_features()`: Mean, variance, skewness, kurtosis

**Testing:**
- Test with synthetic spectrum (known features)
- Test with real Cu K-edge spectrum
- Validate against manual calculations

---

### Task 1.3: Batch Dataset Assembler
**New File:** `xas_ml_modules/xas_batch_assembler.py`

**Implementation:**
```python
class XASBatchAssembler:
    """Assemble individual XAS results into dataset for ML analysis."""
    
    def assemble_dataset(self,
                        sample_results: List[XASSampleResult],
                        quality_filter: str = "usable") -> XASDataset:
        """
        Assemble feature matrix and metadata from sample results.
        
        Parameters
        ----------
        sample_results : list
            List of XASSampleResult objects from preprocessing
        quality_filter : str
            "all", "usable", or "usable_strict" (exclude warnings)
            
        Returns
        -------
        XASDataset
            Feature matrix + metadata ready for ML
        """
        # Filter by quality
        valid_samples = self._filter_by_quality(sample_results, quality_filter)
        
        # Extract features into matrix
        feature_matrix, feature_names = self._build_feature_matrix(valid_samples)
        
        # Extract metadata
        metadata_df = self._build_metadata_table(valid_samples)
        
        # Collect quality flags
        quality_flags = self._collect_quality_flags(valid_samples)
        
        return XASDataset(
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            sample_names=[s.sample_name for s in valid_samples],
            metadata=metadata_df,
            quality_flags=quality_flags,
            n_samples=len(valid_samples),
            n_features=len(feature_names)
        )
```

**Testing:**
- Test with 10 synthetic samples
- Test quality filtering logic
- Verify feature matrix dimensions

---

## Phase 2: ML Core Modules (Week 3-4)

### Task 2.1: PCA Analyzer
**New File:** `xas_ml_modules/xas_pca_analyzer.py`

**Key Functions:**
- `fit_pca()`: Fit PCA to feature matrix
- `determine_n_components()`: Automatic selection (Kaiser, elbow, variance threshold)
- `compute_loadings()`: Feature importance per component
- `transform()`: Project samples to PC space
- `validate_stability()`: Bootstrap validation

**Deliverables:**
- `PCAAnalysisResult` object with all metrics
- Scree plot generation
- Biplot visualization

---

### Task 2.2: Clustering Module
**New File:** `xas_ml_modules/xas_clustering.py`

**Algorithms to implement:**
1. **K-Means** (sklearn.cluster.KMeans)
   - Automatic k selection via silhouette
2. **Hierarchical Ward** (sklearn.cluster.AgglomerativeClustering)
   - Dendrogram plotting
3. **DBSCAN** (optional, Phase 3)
4. **GMM** (optional, Phase 3)

**Key Functions:**
- `cluster()`: Main clustering function
- `select_optimal_k()`: Silhouette-based selection
- `validate_clusters()`: Physical validation (edge consistency, spectral coherence)
- `compute_cluster_representatives()`: Mean spectra per cluster

**Deliverables:**
- `ClusteringResult` object
- Cluster map visualization (adapted from XPS implementation)
- Representative spectra plot

---

### Task 2.3: Trend Analyzer
**New File:** `xas_ml_modules/xas_trend_analyzer.py`

**Key Functions:**
- `correlate_features_metadata()`: Pearson/Spearman correlations
- `detect_outliers()`: Mahalanobis distance in PCA space
- `cluster_metadata_analysis()`: Per-cluster metadata distributions
- `generate_correlation_heatmap()`: Seaborn heatmap

**Deliverables:**
- `TrendAnalysisResult` object
- Correlation heatmap
- Outlier detection plot
- CSV export of significant correlations

---

## Phase 3: Integration & Batch Processing (Week 5-6)

### Task 3.1: Batch Processor
**New File:** `xas_ml_modules/xas_batch_processor.py`

**Main Class:**
```python
class XASBatchProcessor:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.preprocessor = XASAutomatedProcessor()
        self.feature_extractor = XASFeatureExtractor()
        self.assembler = XASBatchAssembler()
        self.pca_analyzer = XASPCAAnalyzer()
        self.clusterer = XASClusterer()
        self.trend_analyzer = XASTrendAnalyzer()
    
    def process_directory(self, data_dir: Path, ...) -> XASDataset:
        """Preprocess all files in parallel."""
        pass
    
    def process_batch_with_ml(self, dataset: XASDataset, ...) -> XASDatasetAnalysis:
        """Run full ML pipeline."""
        pass
    
    def export_results(self, results: XASDatasetAnalysis) -> None:
        """Export JSON, CSV, plots."""
        pass
```

**Parallelization:**
- Use `joblib.Parallel` for preprocessing individual spectra
- Sequential for ML steps (they require full dataset)

---

### Task 3.2: ML Plotting Module
**New File:** `xas_ml_modules/xas_ml_plotter.py`

**Functions:**
- `plot_pca_scree()`
- `plot_pca_biplot()`
- `plot_cluster_map()` (adapt from XPS_mapper)
- `plot_cluster_spectra()`
- `plot_correlation_heatmap()`
- `plot_outlier_scores()`
- `plot_dendrogram()` (hierarchical only)

**Style:**
- Use matplotlib + seaborn
- Consistent color schemes
- Save as PNG (300 dpi)

---

### Task 3.3: Integration with Main Workflow
**File:** `xas_workflow.py`

**Add method to `XASAutomatedProcessor`:**
```python
def process_batch_with_ml(self, 
                         file_list: List[Path],
                         pca_params: Optional[Dict] = None,
                         clustering_params: Optional[Dict] = None) -> XASDatasetAnalysis:
    """
    Process multiple files and run ML analysis.
    
    This is the main entry point for agent workflows.
    """
    # 1. Preprocess all files
    sample_results = [self.process_file(f) for f in file_list]
    
    # 2. Extract features and assemble dataset
    dataset = self.assembler.assemble_dataset(sample_results)
    
    # 3. Run ML pipeline
    ml_results = self._run_ml_pipeline(dataset, pca_params, clustering_params)
    
    # 4. Export all results
    self._export_all_results(ml_results)
    
    return ml_results
```

---

## Phase 4: Testing & Documentation (Week 7)

### Task 4.1: Unit Tests
**Directory:** `tests/test_xas_ml/`

**Test files:**
- `test_feature_extractor.py`
- `test_pca_analyzer.py`
- `test_clustering.py`
- `test_trend_analyzer.py`
- `test_batch_processor.py`

**Test data:**
- Create synthetic XAS dataset (20-50 spectra)
- Use real Cu K-edge data from existing tests
- Include edge cases (noisy data, missing metadata, etc.)

---

### Task 4.2: Integration Test
**File:** `tests/test_xas_ml_integration.py`

**End-to-end test:**
```python
def test_full_ml_pipeline():
    """Test complete workflow: raw data → ML analysis."""
    
    # 1. Create test dataset directory
    test_dir = create_synthetic_xas_dataset(n_samples=30)
    
    # 2. Run batch processor
    processor = XASBatchProcessor(output_dir="./test_output")
    results = processor.process_directory(
        data_dir=test_dir,
        pca_params={"n_components": 3},
        clustering_params={"method": "kmeans", "n_clusters": 4}
    )
    
    # 3. Validate results structure
    assert results.dataset_info.n_samples == 30
    assert results.pca_analysis.n_components == 3
    assert results.clustering.n_clusters == 4
    
    # 4. Check outputs exist
    assert (Path("./test_output") / "features.csv").exists()
    assert (Path("./test_output") / "ml_analysis.json").exists()
    assert (Path("./test_output") / "plots" / "pca_scree.png").exists()
```

---

### Task 4.3: Documentation
**Files to create:**

1. **`xas_ml_modules/README.md`**
   - Overview of ML capabilities
   - Quick start guide
   - API reference

2. **`examples/xas_ml_workflow_example.ipynb`**
   - Jupyter notebook with step-by-step example
   - Uses real or synthetic data
   - Shows all visualizations

3. **`docs/agent_integration_guide.md`**
   - How agents should call the ML pipeline
   - Decision tree for when to use ML vs. simple analysis
   - Interpreting confidence scores and flags

---

## Phase 5 (Future): Advanced Features

### Task 5.1: Simulation Validation (Optional)
**File:** `xas_ml_modules/xas_simulation_validator.py`

- Integrate with xraylarch's FEFF interface
- Generate theoretical spectra for cluster representatives
- Compute R-factor between simulated and experimental

---

### Task 5.2: Supervised Learning (Future)
**File:** `xas_ml_modules/xas_supervised_learning.py`

- Train regression/classification models
- Predict material properties from spectra
- Cross-validation and model selection

---

### Task 5.3: Deep Learning Embeddings (Future)
- Autoencoder for nonlinear dimensionality reduction
- Transfer learning from pre-trained spectroscopy models

---

## File Structure (Target)

```
zzy_llm/Tools/APS_XAS/
├── xas_reader/                    # Existing
│   ├── xas_reader.py
│   └── xas_reference_loader.py
│
├── xas_analyzer/                  # Existing (extend)
│   ├── xas_analyzer_main.py
│   ├── xas_normalization.py
│   ├── xas_models.py              # ← EXTEND with new models
│   └── xas_feature_extractor.py   # ← NEW
│
├── xas_ml_modules/                # NEW directory
│   ├── __init__.py
│   ├── xas_batch_assembler.py     # ← NEW
│   ├── xas_pca_analyzer.py        # ← NEW
│   ├── xas_clustering.py          # ← NEW
│   ├── xas_trend_analyzer.py      # ← NEW
│   ├── xas_batch_processor.py     # ← NEW
│   ├── xas_ml_plotter.py          # ← NEW
│   └── README.md                  # ← NEW
│
├── xas_plotter/                   # Existing (extend)
│   └── xas_ml_plots.py            # ← NEW (or add to xas_ml_plotter.py)
│
├── xas_workflow.py                # Existing (extend)
├── xas_automated_processing_workflow_spec.md    # Existing
├── xas_ml_integration_spec.md     # ← CREATED
└── xas_ml_roadmap.md              # ← CREATED (this file)
```

---

## Dependencies Checklist

### Already Installed (check `requirements.txt`)
- ✓ numpy
- ✓ scipy
- ✓ matplotlib
- ✓ xraylarch

### To Add
- [ ] scikit-learn (>= 1.0)
- [ ] pandas (>= 1.3)
- [ ] seaborn (>= 0.11)
- [ ] joblib (for parallel processing)

**Action:** Update `requirements.txt` or `pyproject.toml`

---

## Success Metrics

### Phase 1 Complete When:
- [ ] All new data models pass validation tests
- [ ] Feature extractor produces 18+ features per spectrum
- [ ] Batch assembler creates valid `XASDataset` from 10+ samples

### Phase 2 Complete When:
- [ ] PCA correctly identifies dominant variance sources
- [ ] Clustering produces interpretable chemical groups
- [ ] Trend analysis detects known correlations in synthetic data

### Phase 3 Complete When:
- [ ] Batch processor handles 50+ spectra in < 5 minutes
- [ ] All diagnostic plots render correctly
- [ ] JSON/CSV exports are agent-readable

### Phase 4 Complete When:
- [ ] All unit tests pass (> 90% coverage)
- [ ] Integration test runs end-to-end successfully
- [ ] Example notebook executes without errors

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **PCA yields low variance explanation** | Include more features (EXAFS, higher-order derivatives); use kernel PCA |
| **Clustering produces no clear groups** | Validate dataset homogeneity; try hierarchical or DBSCAN |
| **Metadata correlations are weak** | Acceptable - report "no significant trends" as valid result |
| **Large datasets are too slow** | Implement incremental PCA; parallelize feature extraction |
| **Integration with existing workflow breaks** | Maintain backward compatibility; make ML analysis optional |

---

## Next Steps (Immediate)

1. **Review this roadmap** with team/stakeholders
2. **Set up development branch**: `feature/xas-ml-integration`
3. **Start Phase 1, Task 1.1**: Extend `XASFeatures` model
4. **Create test data**: Generate synthetic XAS dataset for testing

**Estimated Timeline:** 6-7 weeks for Phases 1-4

---

**End of Roadmap**
