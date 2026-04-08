# XAS Machine Learning Integration Specification

**Purpose:**  
This document extends the **XAS Automated Processing Workflow** with machine learning capabilities for large-scale experimental dataset analysis. It integrates XASDAML concepts (feature extraction, dimensionality reduction, clustering, simulation) while maintaining the scientifically conservative and agent-safe principles of the base workflow.

**Based on:**
- `xas_automated_processing_workflow_spec.md` (preprocessing & validation)
- XASDAML framework (ML-based XAS analysis)
- Existing XPS clustering implementation (`XPS_mapper/cluster_plots.py`)

---

## 1. Extended Design Principles

All principles from the base spec apply, plus:

### Core Modularity Principles
- **Plug-and-Play Architecture**: Each module is standalone with clear inputs/outputs
- **Single Responsibility**: One module = one well-defined task
- **No Hidden Dependencies**: All dependencies explicit in imports and signatures
- **Configuration-Driven**: All human inputs via YAML config files (no hardcoded values)
- **Agent-Friendly**: Simple function signatures, structured outputs, no interactive prompts

### Scientific & Technical Principles
- **Batch-aware**: Must handle 10-1000+ spectra efficiently
- **Feature interpretability**: ML features must map to physical XAS concepts
- **Cluster validation**: Clusters must be validated against spectroscopic principles
- **Simulation integration**: Theoretical spectra are **validation tools**, not replacements for experiment
- **Metadata-aware**: Correlate spectral trends with experimental conditions (temperature, pressure, composition, etc.)

### Maintainability & Debugging
- **Logging at every step**: Use Python logging module (INFO, WARNING, ERROR)
- **Explicit error messages**: "Failed at step X with Y" not "Something went wrong"
- **Checkpointing**: Save intermediate results for debugging
- **Unit testable**: Each module can be tested in isolation

> Machine learning identifies patterns; physics validates their meaning.

---

## 2. Extended Workflow Architecture

### 2.1 Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING STAGE (Per-Spectrum)                │
│  raw data → reference → alignment → deglitching → normalization →  │
│  validation → spectrum QC                                           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   FEATURE EXTRACTION (Per-Spectrum)                 │
│  normalized μ(E) → XASFeatures object (edge, white line, XANES,    │
│  EXAFS, derivatives, moments)                                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   BATCH ASSEMBLY (Dataset Level)                    │
│  Individual spectra → Feature matrix X (n_samples × n_features)    │
│  + Metadata table M (n_samples × n_metadata)                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   ML ANALYSIS MODULES (Dataset Level)               │
│  ┌─────────────────┐  ┌────────────────┐  ┌───────────────────┐   │
│  │ PCA Analysis    │→ │ Clustering     │→ │ Trend Analysis    │   │
│  │ (dim reduction) │  │ (k-means, etc) │  │ (correlations)    │   │
│  └─────────────────┘  └────────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                OPTIONAL: SIMULATION VALIDATION                      │
│  Cluster representatives → FEFF simulation → Compare with cluster  │
│  mean spectra → Validation confidence                              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   AGENT-READABLE OUTPUT                             │
│  JSON report + CSV exports + diagnostic plots                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

**Input to ML Stage:**
- Collection of `XASSampleResult` objects (from preprocessing)
- Each contains `XASFeatures` + `ProcessingMetadata`

**Output from ML Stage:**
- `XASDatasetAnalysis` object with:
  - Feature matrix
  - PCA results
  - Cluster assignments
  - Trend correlations
  - Validation metrics

---

## 3. New Module Specifications

### 3.1 Feature Extraction Module (`xas_feature_extractor`)

**Status:** *Partially implemented in `xas_analyzer/xas_models.py`*

**Responsibilities:**
- Extract interpretable features from normalized XAS spectra
- Compute XANES and EXAFS characteristics
- Calculate spectral derivatives and statistical moments

**Feature Categories:**

**A. Edge Features**
- `e0`: Edge energy (eV)
- `edge_step`: Edge step magnitude
- `edge_slope`: Edge slope (derivative maximum)
- `pre_edge_area`: Integrated pre-edge intensity

**B. XANES Features (Near-Edge)**
- `white_line_intensity`: Peak intensity above edge
- `white_line_energy`: Energy of white line maximum
- `white_line_fwhm`: Full-width at half-maximum
- `xanes_area`: Integrated XANES region (E0 to E0+50 eV)
- `xanes_centroid`: Spectral centroid of XANES

**C. EXAFS Features (Extended)**
- `chi_k_rms`: RMS amplitude of χ(k)
- `ft_peak_r`: First shell distance (Å)
- `ft_peak_amp`: First shell amplitude
- `coordination_number_estimate`: Rough CN from FT amplitude
- `exafs_frequency`: Dominant oscillation frequency

**D. Derivative Features**
- `first_derivative_max`: Maximum of dμ/dE
- `second_derivative_zero`: Zero-crossing of d²μ/dE²
- `derivative_spectrum`: Full dμ/dE array (optional)

**E. Statistical Features**
- `spectral_mean`: Mean μ(E) over normalized range
- `spectral_variance`: Variance of normalized spectrum
- `spectral_skewness`: Asymmetry of distribution
- `spectral_kurtosis`: Tailedness of distribution

**Output Format:**
```python
class XASFeatures(BaseModel):
    # Edge
    e0: float
    edge_step: float
    edge_slope: float
    pre_edge_area: float
    
    # XANES
    white_line_intensity: Optional[float]
    white_line_energy: Optional[float]
    white_line_fwhm: Optional[float]
    xanes_area: float
    xanes_centroid: float
    
    # EXAFS
    chi_k_rms: Optional[float]
    ft_peak_r: Optional[float]
    ft_peak_amp: Optional[float]
    
    # Derivatives
    first_derivative_max: float
    second_derivative_zero: Optional[float]
    
    # Statistics
    spectral_mean: float
    spectral_variance: float
    spectral_skewness: float
    spectral_kurtosis: float
```

**Validation Checks:**
- Feature values must be finite (no NaN, Inf)
- Physical constraints (e.g., edge_step > 0)
- Feature confidence scores based on data quality

**Flags:**
- `low_quality_features`: Derived from poor spectra
- `incomplete_exafs`: k-range too short for reliable EXAFS
- `ambiguous_white_line`: Multiple peaks detected

---

### 3.2 Batch Dataset Assembler (`xas_batch_assembler`)

**Responsibilities:**
- Aggregate individual `XASSampleResult` objects into dataset
- Create feature matrix X (n_samples × n_features)
- Create metadata table M (n_samples × n_metadata)
- Filter out invalid/flagged spectra based on quality criteria

**Inputs:**
- List of `XASSampleResult` objects
- Quality filter criteria (e.g., `classification == "usable"`)

**Outputs:**
```python
class XASDataset(BaseModel):
    feature_matrix: np.ndarray  # (n_samples, n_features)
    feature_names: List[str]
    sample_names: List[str]
    metadata: pd.DataFrame  # (n_samples, n_metadata_cols)
    quality_flags: Dict[str, List[str]]  # Per-sample flags
    n_samples: int
    n_features: int
```

**Quality Filtering:**
- Exclude samples with `classification == "invalid"`
- Optionally exclude `usable_with_warning` based on user setting
- Log all exclusions with reasons

---

### 3.3 PCA Analysis Module (`xas_pca_analyzer`)

**Responsibilities:**
- Perform Principal Component Analysis on feature matrix
- Determine optimal number of components
- Compute loadings (feature importance) and scores (sample projections)
- Generate scree plot and biplot visualizations

**Inputs:**
- `XASDataset` object
- `n_components`: Number of PCs (or "auto" for automatic selection)
- `variance_threshold`: Cumulative variance threshold (default: 0.95)

**Outputs:**
```python
class PCAnalysisResult(BaseModel):
    n_components: int
    explained_variance: List[float]  # Per component
    cumulative_variance: List[float]
    loadings: np.ndarray  # (n_features, n_components)
    scores: np.ndarray  # (n_samples, n_components)
    feature_importance: Dict[str, List[float]]  # Top features per PC
    
    # Validation
    variance_captured: float
    kaiser_criterion: int  # Components with eigenvalue > 1
    confidence: float
```

**Automatic Component Selection:**
1. Kaiser criterion (eigenvalue > 1)
2. Elbow in scree plot
3. Cumulative variance threshold
4. Final choice = minimum of above, bounded by [2, 10]

**Validation Metrics:**
- Reconstruction error
- Stability across bootstrap samples
- Physical interpretability of loadings

**Flags:**
- `low_variance_explained`: First N PCs explain < 80% variance
- `unstable_components`: High bootstrap variance
- `dominated_by_single_feature`: One feature > 90% loading

---

### 3.4 Clustering Module (`xas_clustering`)

**Responsibilities:**
- Cluster spectra in PCA-reduced space or full feature space
- Support multiple algorithms with automatic parameter tuning
- Validate clusters against spectroscopic principles
- Compute cluster statistics and representative spectra

**Algorithms Supported:**
1. **K-Means**: Fast, assumes spherical clusters
2. **Hierarchical (Ward)**: Dendrogram for interpretability
3. **DBSCAN**: Density-based, handles noise
4. **Gaussian Mixture Model**: Probabilistic, soft assignments

**Inputs:**
- `XASDataset` or `PCAAnalysisResult`
- `method`: "kmeans" | "hierarchical" | "dbscan" | "gmm"
- `n_clusters`: Number of clusters (or "auto")
- `use_pca`: Whether to cluster in PCA space (recommended)

**Outputs:**
```python
class ClusteringResult(BaseModel):
    method: str
    n_clusters: int
    labels: List[int]  # Per-sample cluster assignment
    cluster_centers: np.ndarray  # (n_clusters, n_features)
    
    # Cluster characteristics
    cluster_info: List[Dict]  # size, mean_spectrum, std_spectrum, metadata
    
    # Validation metrics
    silhouette_score: float  # [-1, 1], higher better
    davies_bouldin_index: float  # Lower better
    calinski_harabasz_score: float  # Higher better
    
    # Physical validation
    spectral_similarity_within: List[float]  # Per cluster
    spectral_separation_between: float
    
    confidence: float
    flags: List[str]
```

**Automatic Cluster Selection (for k-means, GMM):**
- Test k ∈ [2, min(10, n_samples//5)]
- Compute silhouette score for each k
- Choose k with maximum silhouette (if > 0.3)
- Fallback: elbow method on within-cluster sum of squares

**Physical Validation:**
- **Spectral coherence**: Spectra within cluster should have similar shapes
- **Edge consistency**: E0 variation within cluster < 2 eV
- **Feature consistency**: Key features (white line, FT peak) should align

**Flags:**
- `low_silhouette`: Silhouette < 0.3 (poor clustering)
- `singleton_clusters`: Clusters with < 3 members
- `overlapping_clusters`: Centers too close in feature space
- `inconsistent_edge_positions`: E0 spread > 2 eV within cluster

---

### 3.5 Trend Analysis Module (`xas_trend_analyzer`)

**Responsibilities:**
- Correlate spectral features with experimental metadata
- Identify outliers and anomalies
- Generate correlation heatmaps and trend plots

**Inputs:**
- `XASDataset` (features + metadata)
- `ClusteringResult` (optional)
- Metadata columns to analyze (e.g., "temperature", "pressure", "composition")

**Outputs:**
```python
class TrendAnalysisResult(BaseModel):
    # Feature-metadata correlations
    correlations: Dict[str, Dict[str, float]]  # {feature: {metadata: r}}
    p_values: Dict[str, Dict[str, float]]
    
    # Cluster-metadata associations
    cluster_metadata_stats: Optional[Dict]  # Per-cluster metadata distributions
    
    # Outlier detection
    outlier_indices: List[int]
    outlier_scores: List[float]  # Mahalanobis distance or similar
    
    # Key trends
    significant_correlations: List[Dict]  # |r| > threshold, p < 0.05
    
    confidence: float
    flags: List[str]
```

**Correlation Methods:**
- Pearson correlation for continuous metadata
- Spearman correlation for ranked/ordinal metadata
- ANOVA F-statistic for categorical metadata

**Outlier Detection:**
- Mahalanobis distance in PCA space
- Isolation Forest in full feature space
- Flag samples > 3σ from cluster center

**Flags:**
- `no_significant_correlations`: No feature-metadata pairs with p < 0.05
- `high_outlier_fraction`: > 10% of samples flagged as outliers
- `confounded_metadata`: Multiple metadata strongly correlated

---

### 3.6 Simulation Validation Tool (`xas_simulation_validator`)

**Responsibilities:**
- Generate theoretical XAS spectra using FEFF/xraylarch
- Compare simulated spectra with cluster representatives
- Provide validation confidence scores

**Status:** *Optional - implement later*

**Inputs:**
- Atomic structure files (CIF, XYZ, etc.)
- Element and edge information
- Cluster representative spectra (experimental)

**Outputs:**
```python
class SimulationValidation(BaseModel):
    structure_file: str
    simulated_spectrum: Dict  # energy, mu
    
    # Comparison with experimental
    experimental_cluster_id: int
    spectral_similarity: float  # R-factor or similar
    edge_shift: float  # eV
    
    # FEFF parameters used
    feff_params: Dict
    
    confidence: float
    flags: List[str]
```

**Validation Metrics:**
- R-factor between simulated and experimental
- Edge position agreement (ΔE0 < 1 eV is good)
- XANES shape similarity (correlation coefficient)

**Flags:**
- `poor_agreement`: R-factor > 0.1
- `edge_mismatch`: |ΔE0| > 2 eV
- `simulation_convergence_issue`

---

### 3.7 Batch Processing Framework (`xas_batch_processor`)

**Responsibilities:**
- Process large datasets (100-1000+ spectra) efficiently
- Parallel execution where safe
- Progress tracking and logging
- Incremental results saving

**Implementation Strategy:**
```python
class XASBatchProcessor:
    def process_directory(self, 
                         data_dir: Path,
                         reference_file: Optional[str] = None,
                         n_workers: int = 4) -> XASDataset:
        """
        Process all XAS files in directory.
        
        Steps:
        1. Discover all valid XAS files
        2. Process each file through preprocessing pipeline (parallel)
        3. Extract features from each (parallel)
        4. Assemble into XASDataset
        5. Save intermediate results to checkpoint files
        """
        pass
    
    def process_batch_with_ml(self,
                             dataset: XASDataset,
                             pca_params: Dict,
                             clustering_params: Dict,
                             metadata_columns: List[str]) -> XASDatasetAnalysis:
        """
        Run full ML pipeline on assembled dataset.
        
        Steps:
        1. PCA analysis
        2. Clustering
        3. Trend analysis
        4. Generate all diagnostic plots
        5. Export CSV reports
        """
        pass
```

**Checkpointing:**
- Save preprocessing results after each spectrum: `{output_dir}/preprocessed/{sample_name}.json`
- Save feature matrix: `{output_dir}/features.csv`
- Save final ML results: `{output_dir}/ml_analysis.json`

**Error Handling:**
- Continue processing remaining files if one fails
- Log all failures with traceback
- Emit summary report at end

---

## 4. Extended Output Contract

### 4.1 Dataset-Level JSON Output

```json
{
  "dataset_info": {
    "n_samples": 150,
    "n_features": 18,
    "n_valid_samples": 142,
    "n_excluded": 8,
    "exclusion_reasons": {
      "invalid": 5,
      "usable_with_warning": 3
    }
  },
  
  "pca_analysis": {
    "n_components": 5,
    "explained_variance": [0.45, 0.25, 0.12, 0.08, 0.05],
    "cumulative_variance": [0.45, 0.70, 0.82, 0.90, 0.95],
    "variance_captured": 0.95,
    "confidence": 0.88,
    "flags": []
  },
  
  "clustering": {
    "method": "kmeans",
    "n_clusters": 4,
    "silhouette_score": 0.62,
    "cluster_sizes": [45, 38, 32, 27],
    "cluster_metadata": [
      {
        "cluster_id": 0,
        "size": 45,
        "mean_e0": 7112.3,
        "std_e0": 0.8,
        "dominant_metadata": {"temperature": "300K", "oxidation_state": "Fe2+"}
      }
    ],
    "confidence": 0.75,
    "flags": []
  },
  
  "trend_analysis": {
    "significant_correlations": [
      {
        "feature": "white_line_intensity",
        "metadata": "temperature",
        "r": -0.72,
        "p_value": 0.0001,
        "interpretation": "White line intensity decreases with temperature"
      }
    ],
    "outlier_count": 8,
    "outlier_indices": [12, 45, 67, ...],
    "confidence": 0.81
  },
  
  "validation_summary": {
    "overall_quality": "good",
    "preprocessing_success_rate": 0.95,
    "ml_analysis_confidence": 0.78,
    "recommended_actions": [
      "Cluster 3 has low spectral coherence - consider sub-clustering",
      "Temperature correlation is strong - include in downstream modeling"
    ]
  }
}
```

### 4.2 CSV Exports

**`features.csv`**: Feature matrix with sample names and metadata
```csv
sample_name,e0,edge_step,white_line_intensity,xanes_area,...,temperature,pressure,cluster_id
Fe_sample_001,7112.3,0.95,1.23,45.2,...,300,1.0,0
Fe_sample_002,7112.5,0.88,1.18,43.8,...,400,1.0,1
...
```

**`cluster_summary.csv`**: Cluster statistics
```csv
cluster_id,size,mean_e0,std_e0,mean_white_line,silhouette_contribution
0,45,7112.3,0.8,1.23,0.65
1,38,7113.1,1.2,1.08,0.58
...
```

**`correlations.csv`**: Feature-metadata correlations
```csv
feature,metadata,correlation,p_value,significant
white_line_intensity,temperature,-0.72,0.0001,True
edge_step,pressure,0.15,0.08,False
...
```

---

## 5. Diagnostic Plots

### 5.1 Required Plots

1. **PCA Scree Plot** (`pca_scree.png`)
   - Explained variance per component
   - Cumulative variance line
   - Kaiser criterion threshold

2. **PCA Biplot** (`pca_biplot.png`)
   - PC1 vs PC2 scatter (colored by cluster if available)
   - Feature loading vectors
   - Sample labels for outliers

3. **Cluster Map** (`cluster_spatial_map.png`) *[if spatial coords available]*
   - Spatial distribution of clusters
   - Adapted from XPS implementation

4. **Cluster Spectra** (`cluster_representative_spectra.png`)
   - Mean spectrum per cluster ± standard deviation
   - Energy axis inverted (spectroscopy convention)

5. **Correlation Heatmap** (`feature_metadata_correlations.png`)
   - Features × metadata
   - Color-coded by correlation strength
   - Significance markers (p < 0.05)

6. **Outlier Scores** (`outlier_detection.png`)
   - Histogram of outlier scores
   - Threshold line
   - Flagged samples highlighted

### 5.2 Optional Plots

7. **Dendrogram** (`hierarchical_dendrogram.png`) *[if hierarchical clustering]*
8. **Silhouette Plot** (`silhouette_analysis.png`)
9. **Feature Importance** (`pca_feature_loadings.png`)
10. **Metadata Distributions per Cluster** (`cluster_metadata_boxplots.png`)

---

## 6. Integration with Agent Workflow

### 6.1 Agent Decision Tree

```
Agent receives: "Analyze XAS dataset at /path/to/data/"

├─ Step 1: Batch Preprocessing
│  ├─ Call XASBatchProcessor.process_directory()
│  ├─ Check: preprocessing_success_rate > 0.8?
│  │  ├─ Yes → Continue
│  │  └─ No → Flag low-quality dataset, recommend manual inspection
│
├─ Step 2: Feature Extraction & Assembly
│  ├─ Extract features from all valid samples
│  ├─ Create XASDataset
│  ├─ Check: n_valid_samples > 20?
│  │  ├─ Yes → ML analysis viable
│  │  └─ No → Skip ML, only descriptive statistics
│
├─ Step 3: ML Analysis (if viable)
│  ├─ PCA: Determine dimensionality
│  ├─ Clustering: Identify spectral groups
│  ├─ Trends: Correlate with metadata
│  ├─ Check: ml_analysis_confidence > 0.7?
│  │  ├─ Yes → Results are reliable
│  │  └─ No → Flag as exploratory, recommend more data
│
├─ Step 4: Optional Simulation (if structures provided)
│  ├─ Generate FEFF spectra for cluster representatives
│  ├─ Validate against experimental
│
└─ Step 5: Report Generation
   ├─ Generate all diagnostic plots
   ├─ Export CSV files
   ├─ Create JSON summary
   ├─ Generate human-readable summary with recommendations
```

### 6.2 Agent-Readable Recommendations

The ML modules should emit **actionable recommendations** for the agent:

```json
"agent_recommendations": [
  {
    "priority": "high",
    "category": "data_quality",
    "message": "15 samples have low edge step - consider removing or flagging",
    "action": "filter_samples",
    "parameters": {"quality_threshold": "usable"}
  },
  {
    "priority": "medium",
    "category": "analysis",
    "message": "Strong temperature dependence detected - include in predictive model",
    "action": "add_feature",
    "parameters": {"feature": "temperature"}
  },
  {
    "priority": "low",
    "category": "clustering",
    "message": "Cluster 2 may benefit from sub-clustering (high intra-cluster variance)",
    "action": "refine_clustering",
    "parameters": {"cluster_id": 2, "method": "hierarchical"}
  }
]
```

---

## 7. Dependencies

### 7.1 Core Dependencies (Already in base spec)
- `numpy`
- `scipy`
- `matplotlib`
- `xraylarch`

### 7.2 Additional Dependencies for ML
- `scikit-learn` (PCA, clustering, metrics)
- `pandas` (DataFrames for metadata)
- `seaborn` (enhanced plotting)
- `joblib` (parallel processing)

### 7.3 Optional Dependencies
- `tensorflow` / `keras` (for future deep learning)
- `pymatgen` (structure handling for simulation)
- `pydot` (for dendrogram visualization)

---

## 8. Validation & Safety Rules

### 8.1 ML-Specific Safety Rules

1. **Minimum Sample Size**: 
   - PCA requires n_samples ≥ 20 (warn if < 50)
   - Clustering requires n_samples ≥ 3 × n_clusters

2. **Feature Scaling**:
   - Always standardize features before PCA/clustering
   - Document scaling parameters in output

3. **Overfitting Prevention**:
   - Do NOT use clustering labels as features for predictive models
   - Cross-validate any supervised learning (future)

4. **Physical Constraints**:
   - Clusters must respect spectroscopic principles (edge consistency)
   - Reject clusters with physically impossible feature combinations

5. **Confidence Reporting**:
   - Every ML result must have confidence score
   - Low confidence (< 0.6) triggers warning flag

### 8.2 Failure Modes

| Failure | Cause | Agent Response |
|---------|-------|----------------|
| `insufficient_variance` | First 10 PCs explain < 50% variance | Flag: "Features may be redundant or noisy" |
| `clustering_unstable` | Silhouette < 0.2 | Flag: "Clusters not well-separated, use with caution" |
| `no_metadata_correlation` | All p-values > 0.05 | Flag: "No significant metadata trends detected" |
| `outlier_dominated` | > 30% samples are outliers | Flag: "Dataset may be heterogeneous or low quality" |

---

## 9. Implementation Priority

### Phase 1 (Immediate)
1. ✅ Extend `XASFeatures` model with all feature categories
2. ✅ Implement `XASBatchAssembler`
3. ✅ Implement `XASPCAAnalyzer`
4. ✅ Create CSV export functions

### Phase 2 (Short-term)
5. Implement `XASClustering` (k-means, hierarchical)
6. Implement `XASTrendAnalyzer`
7. Create all diagnostic plotting functions
8. Integrate with existing `XASAutomatedProcessor`

### Phase 3 (Medium-term)
9. Implement `XASBatchProcessor` with parallelization
10. Add DBSCAN and GMM clustering
11. Optimize for large datasets (> 500 samples)

### Phase 4 (Long-term)
12. Implement `XASSimulationValidator` with FEFF
13. Add supervised learning capabilities (predict properties from spectra)
14. Integrate with broader `zzy_llm` agent workflow

---

## 10. Example Usage

### 10.1 Basic Batch Analysis

```python
from zzy_llm.Tools.APS_XAS import XASBatchProcessor

# Process directory of XAS files
processor = XASBatchProcessor(output_dir="./results")
dataset = processor.process_directory(
    data_dir="./raw_data/Fe_temperature_series/",
    reference_file="./references/Fe_foil.xdi",
    n_workers=4
)

# Run ML analysis
ml_results = processor.process_batch_with_ml(
    dataset=dataset,
    pca_params={"n_components": "auto"},
    clustering_params={"method": "kmeans", "n_clusters": "auto"},
    metadata_columns=["temperature", "pressure"]
)

# Results automatically saved to:
# - results/features.csv
# - results/ml_analysis.json
# - results/plots/*.png
```

### 10.2 Agent Workflow Integration

```python
# Agent receives task
task = {
    "type": "analyze_xas_dataset",
    "data_dir": "/path/to/data",
    "metadata_file": "experimental_conditions.csv",
    "reference": "Fe_foil.xdi"
}

# Agent calls batch processor
results = xas_batch_workflow(
    data_dir=task["data_dir"],
    reference=task["reference"],
    metadata_file=task["metadata_file"]
)

# Agent reads structured output
if results["validation_summary"]["overall_quality"] == "good":
    if results["trend_analysis"]["significant_correlations"]:
        agent.recommend("Include temperature as feature in predictive model")
    
    if results["clustering"]["confidence"] > 0.7:
        agent.report(f"Identified {results['clustering']['n_clusters']} distinct chemical states")
else:
    agent.flag("Dataset requires manual review - low preprocessing success rate")
```

---

## 11. Deliverables

1. **Python Modules**
   - `xas_feature_extractor.py` (extend existing `xas_models.py`)
   - `xas_batch_assembler.py`
   - `xas_pca_analyzer.py`
   - `xas_clustering.py`
   - `xas_trend_analyzer.py`
   - `xas_batch_processor.py`
   - `xas_ml_plotter.py` (ML-specific plots)

2. **Updated Models**
   - Extended `XASFeatures` class
   - New `XASDataset`, `PCAAnalysisResult`, `ClusteringResult`, `TrendAnalysisResult` classes

3. **Documentation**
   - API documentation for all new modules
   - Jupyter notebook with example workflow
   - Agent integration guide

4. **Tests**
   - Unit tests for each module
   - Integration test with synthetic dataset (10-50 spectra)
   - End-to-end test with real XAS data

---

**End of ML Integration Specification**
