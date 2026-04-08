# XAS ML Integration: Quick Reference

**Last Updated:** March 3, 2026  
**For:** Developers implementing the ML integration

---

## 📋 Planning Documents Created

| Document | Purpose | Read When... |
|----------|---------|--------------|
| **xas_ml_integration_spec.md** | Detailed technical specification | Implementing any module |
| **xas_ml_roadmap.md** | 4-phase implementation plan | Planning work, tracking progress |
| **xas_ml_planning_summary.md** | Executive summary | First time overview, stakeholder review |
| **xas_architecture_diagram.md** | Visual workflow diagrams | Understanding data flow, debugging |
| **quick_reference.md** | This file | Quick lookups during development |

---

## 🎯 Quick Start: What to Implement First

### Phase 1 (Foundation - Weeks 1-2)
```
1. xas_analyzer/xas_models.py           # Extend XASFeatures, add new models
2. xas_analyzer/xas_feature_extractor.py # Feature extraction logic
3. xas_ml_modules/xas_batch_assembler.py # Dataset assembly
```

### Phase 2 (ML Core - Weeks 3-4)
```
4. xas_ml_modules/xas_pca_analyzer.py    # PCA implementation
5. xas_ml_modules/xas_clustering.py      # Clustering algorithms
6. xas_ml_modules/xas_trend_analyzer.py  # Correlation analysis
```

### Phase 3 (Integration - Weeks 5-6)
```
7. xas_ml_modules/xas_batch_processor.py # Orchestrator
8. xas_ml_modules/xas_ml_plotter.py      # Visualization
9. Update xas_workflow.py                # Main entry point
```

---

## 📊 Key Data Models

### XASFeatures (18+ features)
```python
class XASFeatures(BaseModel):
    # Edge (4)
    e0: float
    edge_step: float
    edge_slope: float
    pre_edge_area: float
    
    # XANES (5)
    white_line_intensity: Optional[float]
    white_line_energy: Optional[float]
    white_line_fwhm: Optional[float]
    xanes_area: float
    xanes_centroid: float
    
    # EXAFS (3)
    chi_k_rms: Optional[float]
    ft_peak_r: Optional[float]
    ft_peak_amp: Optional[float]
    
    # Derivatives (2)
    first_derivative_max: float
    second_derivative_zero: Optional[float]
    
    # Statistics (4)
    spectral_mean: float
    spectral_variance: float
    spectral_skewness: float
    spectral_kurtosis: float
```

### XASDataset (batch container)
```python
class XASDataset(BaseModel):
    feature_matrix: np.ndarray      # (n_samples, n_features)
    feature_names: List[str]        # 18+ feature names
    sample_names: List[str]         # Sample identifiers
    metadata: pd.DataFrame          # (n_samples, n_metadata_cols)
    quality_flags: Dict             # Per-sample flags
    n_samples: int
    n_features: int
```

### ClusteringResult
```python
class ClusteringResult(BaseModel):
    method: str                     # "kmeans", "hierarchical", etc.
    n_clusters: int
    labels: List[int]               # Per-sample cluster ID
    cluster_centers: np.ndarray
    cluster_info: List[Dict]        # Per-cluster statistics
    silhouette_score: float         # Quality metric [-1, 1]
    confidence: float               # Overall confidence [0, 1]
    flags: List[str]                # Validation warnings
```

---

## 🔧 Core Functions to Implement

### Feature Extraction
```python
# In xas_analyzer/xas_feature_extractor.py
class XASFeatureExtractor:
    def extract_features(self, energy, mu, processing_result) -> XASFeatures:
        """Extract all 18+ features from normalized spectrum."""
        pass
    
    def _compute_edge_slope(self, energy, mu) -> float:
        """Max of first derivative near E0."""
        pass
    
    def _extract_xanes_features(self, energy, mu, e0) -> Dict:
        """White line detection, XANES area, centroid."""
        pass
    
    def _extract_derivative_features(self, energy, mu) -> Dict:
        """1st and 2nd derivative analysis."""
        pass
```

### PCA Analysis
```python
# In xas_ml_modules/xas_pca_analyzer.py
class XASPCAAnalyzer:
    def fit_pca(self, dataset: XASDataset, n_components='auto') -> PCAAnalysisResult:
        """Fit PCA to feature matrix."""
        pass
    
    def determine_n_components(self, X) -> int:
        """Auto-select using Kaiser + elbow + variance threshold."""
        pass
    
    def validate_stability(self, X, n_bootstrap=100) -> float:
        """Bootstrap validation of PCA stability."""
        pass
```

### Clustering
```python
# In xas_ml_modules/xas_clustering.py
class XASClusterer:
    def cluster(self, dataset, method='kmeans', n_clusters='auto') -> ClusteringResult:
        """Main clustering function."""
        pass
    
    def select_optimal_k(self, X, k_range) -> int:
        """Silhouette-based k selection."""
        pass
    
    def validate_clusters(self, dataset, labels) -> Dict:
        """Physical validation: edge consistency, spectral coherence."""
        pass
```

---

## 🧪 Testing Strategy

### Unit Tests (per module)
```python
# tests/test_xas_ml/test_feature_extractor.py
def test_extract_edge_features():
    energy, mu = create_synthetic_spectrum()
    extractor = XASFeatureExtractor()
    features = extractor.extract_features(energy, mu, {})
    assert features.e0 > 0
    assert 0 < features.edge_step < 2

# tests/test_xas_ml/test_pca_analyzer.py
def test_pca_automatic_selection():
    dataset = create_synthetic_dataset(n_samples=50)
    analyzer = XASPCAAnalyzer()
    result = analyzer.fit_pca(dataset, n_components='auto')
    assert 2 <= result.n_components <= 10
```

### Integration Test
```python
# tests/test_xas_ml/test_ml_integration.py
def test_full_pipeline():
    """End-to-end: raw data → ML analysis."""
    processor = XASBatchProcessor(output_dir="./test_output")
    results = processor.process_directory(
        data_dir="./test_data/Fe_series/",
        pca_params={"n_components": 3},
        clustering_params={"method": "kmeans"}
    )
    assert results.clustering.n_clusters > 0
    assert (Path("./test_output") / "features.csv").exists()
```

---

## 📈 Validation Thresholds

### PCA Quality
- ✓ Good: Variance explained ≥ 0.80 (80%)
- ⚠ Acceptable: 0.60 ≤ variance < 0.80
- ❌ Poor: variance < 0.60

### Clustering Quality
- ✓ Good: Silhouette ≥ 0.50
- ⚠ Acceptable: 0.30 ≤ silhouette < 0.50
- ❌ Poor: silhouette < 0.30

### Physical Validation (per cluster)
- ✓ Edge consistency: σ(E0) < 2 eV
- ✓ Spectral coherence: r > 0.80
- ✓ Min cluster size: ≥ 3 samples

### Dataset Requirements
- PCA: n_samples ≥ 20 (warn if < 50)
- Clustering: n_samples ≥ 3 × n_clusters

---

## 🎨 Visualization Requirements

### Required Plots (auto-generate)
1. **pca_scree.png** - Variance per component + cumulative
2. **pca_biplot.png** - PC1 vs PC2 with feature vectors
3. **cluster_map.png** - Spatial distribution (if coords available)
4. **cluster_spectra.png** - Mean ± std per cluster
5. **correlation_heatmap.png** - Features × metadata
6. **outlier_scores.png** - Histogram with threshold

### Plot Style Guidelines
- DPI: 300
- Format: PNG
- Color schemes: 'tab10' for discrete, 'RdBu_r' for diverging
- Font sizes: Title=15, Labels=13, Ticks=11
- Always invert x-axis for binding energy plots

---

## 🚨 Common Pitfalls & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| **Low PCA variance** | Features are redundant | Add EXAFS features, try kernel PCA |
| **Poor clustering** | Dataset too homogeneous | Expected - report "no distinct groups" |
| **NaN in features** | Missing EXAFS data | Make EXAFS features optional, handle gracefully |
| **Slow batch processing** | Sequential execution | Parallelize preprocessing with joblib |
| **Memory errors (large datasets)** | Loading all spectra at once | Use chunked processing, incremental PCA |

---

## 📦 Dependencies Checklist

```bash
# Core (already have)
pip install numpy scipy matplotlib xraylarch

# ML integration (need to add)
pip install scikit-learn>=1.0 pandas>=1.3 seaborn>=0.11 joblib>=1.1

# Optional
pip install pymatgen  # For simulation validation (Phase 4+)
```

---

## 🔗 Key File Locations

### Existing Code (extend/reuse)
```
xas_analyzer/xas_models.py              # Extend XASFeatures here
xas_plotter/quality_control_plotter.py  # Plotting patterns to follow
XPS_mapper/cluster_plots.py             # Clustering viz reference
```

### New Code (create)
```
xas_ml_modules/                         # All new ML code goes here
tests/test_xas_ml/                      # All ML tests
examples/xas_ml_workflow_example.ipynb  # Example notebook
```

---

## 💡 Pro Tips

1. **Always standardize features** before PCA/clustering:
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Handle optional EXAFS gracefully**:
   ```python
   if chi_k_data is not None:
       features['chi_k_rms'] = compute_rms(chi_k_data)
   else:
       features['chi_k_rms'] = None  # Will be excluded from matrix
   ```

3. **Use confidence scores everywhere**:
   ```python
   confidence = min(
       silhouette_score_normalized,
       edge_consistency_score,
       cluster_separation_score
   )
   ```

4. **Leverage existing patterns**:
   - Copy plotting style from `XPS_mapper/cluster_plots.py`
   - Copy data model patterns from `xas_analyzer/xas_models.py`
   - Copy parallel processing from similar tools

5. **Document edge cases in tests**:
   ```python
   def test_clustering_with_missing_exafs():
       """Clustering should work even if EXAFS features are None."""
       dataset = create_dataset_without_exafs()
       result = cluster(dataset)
       assert result.n_clusters > 0
   ```

---

## 📞 Getting Help

- **Spec unclear?** → Read `xas_ml_integration_spec.md` Section X
- **Architecture confusing?** → See `xas_architecture_diagram.md`
- **Timeline questions?** → Check `xas_ml_roadmap.md`
- **Quick overview?** → Read `xas_ml_planning_summary.md`

---

## ✅ Implementation Checklist

### Phase 1
- [ ] Extend `XASFeatures` in `xas_models.py`
- [ ] Create `XASDataset`, `PCAAnalysisResult`, `ClusteringResult` models
- [ ] Implement `XASFeatureExtractor` class
- [ ] Write unit tests for feature extraction
- [ ] Implement `XASBatchAssembler` class
- [ ] Test batch assembly with 10 samples

### Phase 2
- [ ] Implement `XASPCAAnalyzer` class
- [ ] Add automatic component selection
- [ ] Implement `XASClusterer` (k-means, hierarchical)
- [ ] Add cluster physical validation
- [ ] Implement `XASTrendAnalyzer` class
- [ ] Test full ML pipeline with synthetic data

### Phase 3
- [ ] Implement `XASBatchProcessor` orchestrator
- [ ] Add parallel processing support
- [ ] Create `XASMLPlotter` class
- [ ] Implement all required plots
- [ ] Integrate with `xas_workflow.py`
- [ ] Test with real XAS dataset (50+ spectra)

### Phase 4
- [ ] Write comprehensive unit tests (>90% coverage)
- [ ] Create integration test suite
- [ ] Write example Jupyter notebook
- [ ] Document API in README files
- [ ] Conduct code review
- [ ] Merge to main branch

---

**End of Quick Reference**
