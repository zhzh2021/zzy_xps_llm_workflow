# XAS ML Integration: Architecture Diagram

This document provides visual representations of the integrated workflow.

---

## 1. High-Level Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    XAS AUTOMATED ANALYSIS SYSTEM                         ║
║                  (Experimental Data → Insights)                          ║
╚══════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────┐
│  INPUT: Raw Experimental XAS Data                                        │
│  • Beamline files (XDI, ASCII, HDF5)                                     │
│  • Reference spectra (foils, standards)                                  │
│  • Metadata (temperature, pressure, composition, ...)                    │
└──────────────────────────────────────────────────────────────────────────┘
                                   ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: PER-SPECTRUM PREPROCESSING (Existing)                          │
│  ┌────────┐  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐       │
│  │ Reader │→ │ Align   │→ │ Deglitch │→ │  Norm  │→ │   QC     │       │
│  │ (XDI)  │  │ (E0)    │  │ (spikes) │  │ (μ→χ)  │  │ (valid?) │       │
│  └────────┘  └─────────┘  └──────────┘  └────────┘  └──────────┘       │
│                                                                           │
│  Output: XASSampleResult (normalized spectrum + metadata + QC flags)     │
└──────────────────────────────────────────────────────────────────────────┘
                                   ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: FEATURE EXTRACTION (New)                                       │
│  ┌─────────────────────────────────────────────────────────────┐         │
│  │  XASFeatureExtractor                                        │         │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │         │
│  │  │   Edge   │  │  XANES   │  │  EXAFS   │  │Derivatives│  │         │
│  │  │ Features │  │ Features │  │ Features │  │& Stats    │  │         │
│  │  └──────────┘  └──────────┘  └──────────┘  └───────────┘  │         │
│  └─────────────────────────────────────────────────────────────┘         │
│                                                                           │
│  Output: XASFeatures (18+ numerical descriptors per spectrum)            │
└──────────────────────────────────────────────────────────────────────────┘
                                   ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: BATCH ASSEMBLY (New)                                           │
│  ┌────────────────────────────────────────────────────────┐              │
│  │  XASBatchAssembler                                     │              │
│  │  • Collect all XASFeatures into matrix X (n × p)      │              │
│  │  • Collect metadata into DataFrame M (n × m)          │              │
│  │  • Filter by quality (remove "invalid" samples)       │              │
│  └────────────────────────────────────────────────────────┘              │
│                                                                           │
│  Output: XASDataset (feature matrix + metadata + sample names)           │
└──────────────────────────────────────────────────────────────────────────┘
                                   ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: ML ANALYSIS PIPELINE (New)                                     │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐            │
│  │  Step 4.1: Dimensionality Reduction (PCA)               │            │
│  │  • Auto-select n_components (Kaiser, elbow, variance)   │            │
│  │  • Compute loadings (feature importance)                │            │
│  │  • Project to PC space (scores)                         │            │
│  └──────────────────────────────────────────────────────────┘            │
│                          ↓                                               │
│  ┌──────────────────────────────────────────────────────────┐            │
│  │  Step 4.2: Clustering                                    │            │
│  │  • Choose algorithm (k-means, hierarchical, DBSCAN)     │            │
│  │  • Auto-select k (silhouette maximization)              │            │
│  │  • Validate clusters (edge consistency, coherence)      │            │
│  │  • Compute cluster representatives (mean spectra)       │            │
│  └──────────────────────────────────────────────────────────┘            │
│                          ↓                                               │
│  ┌──────────────────────────────────────────────────────────┐            │
│  │  Step 4.3: Trend Analysis                                │            │
│  │  • Correlate features with metadata (Pearson/Spearman)  │            │
│  │  • Detect outliers (Mahalanobis distance)               │            │
│  │  • Per-cluster metadata statistics                      │            │
│  └──────────────────────────────────────────────────────────┘            │
│                          ↓                                               │
│  ┌──────────────────────────────────────────────────────────┐            │
│  │  Step 4.4: Simulation Validation (Optional)             │            │
│  │  • Generate FEFF spectra for cluster representatives    │            │
│  │  • Compare with experimental (R-factor)                 │            │
│  └──────────────────────────────────────────────────────────┘            │
│                                                                           │
│  Output: XASDatasetAnalysis (PCA + Clustering + Trends + Validation)     │
└──────────────────────────────────────────────────────────────────────────┘
                                   ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: Multi-Format Results                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ JSON Report  │  │  CSV Exports │  │    Plots     │                   │
│  │ (agent-safe) │  │  (features,  │  │ (PCA, clust, │                   │
│  │              │  │  correlations)│  │  heatmaps)   │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Model Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│  XASSampleResult (per-spectrum output from preprocessing)       │
│  ┌──────────────────────────────────────────────────┐           │
│  │  sample_name: str                                │           │
│  │  timestamp: datetime                             │           │
│  │  source_file: str                                │           │
│  │  ┌────────────────────────────────────────────┐  │           │
│  │  │  features: XASFeatures                     │  │           │
│  │  │    • e0, edge_step, edge_slope             │  │           │
│  │  │    • white_line_intensity, xanes_area      │  │           │
│  │  │    • chi_k_rms, ft_peak_r                  │  │           │
│  │  │    • derivatives, statistics               │  │           │
│  │  └────────────────────────────────────────────┘  │           │
│  │  ┌────────────────────────────────────────────┐  │           │
│  │  │  processing_metadata: ProcessingMetadata   │  │           │
│  │  │    • beamline, element, edge               │  │           │
│  │  │    • temperature, pressure, ...            │  │           │
│  │  └────────────────────────────────────────────┘  │           │
│  │  quality_classification: str                     │           │
│  │  flags: List[str]                                │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              ↓ (assemble n samples)
┌─────────────────────────────────────────────────────────────────┐
│  XASDataset (batch-level container)                             │
│  ┌──────────────────────────────────────────────────┐           │
│  │  feature_matrix: ndarray (n_samples, n_features) │           │
│  │  feature_names: List[str]                        │           │
│  │  sample_names: List[str]                         │           │
│  │  metadata: DataFrame (n_samples, n_metadata)     │           │
│  │  quality_flags: Dict[str, List[str]]             │           │
│  │  n_samples: int                                  │           │
│  │  n_features: int                                 │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              ↓ (apply ML pipeline)
┌─────────────────────────────────────────────────────────────────┐
│  XASDatasetAnalysis (complete ML results)                       │
│  ┌──────────────────────────────────────────────────┐           │
│  │  dataset_info: Dict                              │           │
│  │  ┌────────────────────────────────────────────┐  │           │
│  │  │  pca_analysis: PCAAnalysisResult           │  │           │
│  │  │    • n_components, explained_variance      │  │           │
│  │  │    • loadings, scores                      │  │           │
│  │  │    • confidence, flags                     │  │           │
│  │  └────────────────────────────────────────────┘  │           │
│  │  ┌────────────────────────────────────────────┐  │           │
│  │  │  clustering: ClusteringResult              │  │           │
│  │  │    • method, n_clusters, labels            │  │           │
│  │  │    • cluster_centers, cluster_info         │  │           │
│  │  │    • silhouette_score, confidence          │  │           │
│  │  └────────────────────────────────────────────┘  │           │
│  │  ┌────────────────────────────────────────────┐  │           │
│  │  │  trend_analysis: TrendAnalysisResult       │  │           │
│  │  │    • correlations, p_values                │  │           │
│  │  │    • outlier_indices, outlier_scores       │  │           │
│  │  │    • significant_correlations              │  │           │
│  │  └────────────────────────────────────────────┘  │           │
│  │  validation_summary: Dict                        │           │
│  │  agent_recommendations: List[Dict]               │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Dependencies

```
┌──────────────────────────────────────────────────────────────┐
│  xas_workflow.py                                             │
│  (Main orchestrator)                                         │
└──────────────────────────────────────────────────────────────┘
          ↓ uses                  ↓ uses                ↓ uses
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  xas_reader/     │   │  xas_analyzer/   │   │  xas_plotter/    │
│  • xas_reader.py │   │  • normalization │   │  • diagnostics   │
│  • reference.py  │   │  • validation    │   │  • quality plots │
└──────────────────┘   │  • QC            │   └──────────────────┘
                       └──────────────────┘
                               ↓ extends
                       ┌──────────────────────────────────────┐
                       │  xas_analyzer/                       │
                       │  • xas_models.py (extended)          │
                       │  • xas_feature_extractor.py (NEW)    │
                       └──────────────────────────────────────┘
                               ↓ produces
                       ┌──────────────────┐
                       │  XASFeatures     │
                       └──────────────────┘
                               ↓ consumed by
┌───────────────────────────────────────────────────────────────────┐
│  xas_ml_modules/ (NEW)                                            │
│  ┌──────────────────────┐  ┌──────────────────────┐              │
│  │ xas_batch_assembler  │→ │  xas_pca_analyzer    │              │
│  │ (XASDataset builder) │  │  (PCA, loadings)     │              │
│  └──────────────────────┘  └──────────────────────┘              │
│                                      ↓                            │
│  ┌──────────────────────┐  ┌──────────────────────┐              │
│  │  xas_clustering      │← │ xas_trend_analyzer   │              │
│  │  (k-means, hier, ..) │  │ (correlations, etc)  │              │
│  └──────────────────────┘  └──────────────────────┘              │
│                                      ↓                            │
│  ┌──────────────────────────────────────────────────┐            │
│  │  xas_batch_processor (orchestrates all above)    │            │
│  └──────────────────────────────────────────────────┘            │
│                                      ↓                            │
│  ┌──────────────────────────────────────────────────┐            │
│  │  xas_ml_plotter (visualization suite)            │            │
│  └──────────────────────────────────────────────────┘            │
└───────────────────────────────────────────────────────────────────┘
                               ↓ produces
                       ┌──────────────────┐
                       │ XASDatasetAnalysis│
                       └──────────────────┘
```

---

## 4. Feature Extraction Detail

```
┌────────────────────────────────────────────────────────────────┐
│  Normalized XAS Spectrum (μ(E))                                │
│                                                                 │
│   μ ↑                                                           │
│     │        ╱──────  ← post-edge                              │
│   1 │       ╱│                                                  │
│     │      ╱ │ white line                                      │
│     │     ╱  ▲                                                  │
│     │    ╱   │                                                  │
│   0 │───╱────┼────→ E                                          │
│     │   ↑    E0                                                │
│     └───┴────────────────────────────────────                  │
│       pre-edge                                                 │
└────────────────────────────────────────────────────────────────┘
                        ↓ extract
┌────────────────────────────────────────────────────────────────┐
│  XASFeatures Object                                            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  EDGE FEATURES                                      │       │
│  │  • e0 = 7112.3 eV              (edge position)      │       │
│  │  • edge_step = 0.95            (step height)        │       │
│  │  • edge_slope = 0.45           (max of dμ/dE)       │       │
│  │  • pre_edge_area = 0.12        (integrated area)    │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  XANES FEATURES (E0 to E0+50 eV)                    │       │
│  │  • white_line_intensity = 1.23  (peak above edge)   │       │
│  │  • white_line_energy = 7115.1   (peak position)     │       │
│  │  • white_line_fwhm = 3.2        (peak width)        │       │
│  │  • xanes_area = 45.2            (total area)        │       │
│  │  • xanes_centroid = 7118.5      (spectral center)   │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  EXAFS FEATURES (k-space, R-space)                  │       │
│  │  • chi_k_rms = 0.82             (oscillation amp)   │       │
│  │  • ft_peak_r = 1.95 Å           (1st shell dist)    │       │
│  │  • ft_peak_amp = 3.4            (coordination amp)  │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  DERIVATIVE FEATURES                                │       │
│  │  • first_derivative_max = 0.45  (at E0)             │       │
│  │  • second_derivative_zero = 7113.2 (inflection)     │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  STATISTICAL FEATURES                               │       │
│  │  • spectral_mean = 0.52         (avg intensity)     │       │
│  │  • spectral_variance = 0.18     (spread)            │       │
│  │  • spectral_skewness = 0.34     (asymmetry)         │       │
│  │  • spectral_kurtosis = -0.21    (tailedness)        │       │
│  └─────────────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. Clustering Validation Logic

```
┌──────────────────────────────────────────────────────────────┐
│  Clustering Result (n_clusters = 4 proposed)                 │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│  VALIDATION CHECKS                                           │
│                                                               │
│  ┌────────────────────────────────────────────────┐          │
│  │  1. Mathematical Validation                    │          │
│  │     ✓ Silhouette score = 0.62 (> 0.3 threshold)│          │
│  │     ✓ Davies-Bouldin = 0.85 (< 1.0 is good)    │          │
│  │     ✓ No singleton clusters                    │          │
│  └────────────────────────────────────────────────┘          │
│                                                               │
│  ┌────────────────────────────────────────────────┐          │
│  │  2. Physical Validation (XAS-specific)         │          │
│  │     For each cluster:                          │          │
│  │     ✓ E0 std dev < 2 eV (edge consistency)     │          │
│  │     ✓ Spectral similarity > 0.8 (coherence)    │          │
│  │     ✓ White line positions aligned             │          │
│  └────────────────────────────────────────────────┘          │
│                                                               │
│  ┌────────────────────────────────────────────────┐          │
│  │  3. Metadata Consistency Check                 │          │
│  │     - Cluster 0: 90% samples at T=300K ✓       │          │
│  │     - Cluster 1: Mixed temperatures ⚠          │          │
│  │     → Flag cluster 1 for review                │          │
│  └────────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│  DECISION                                                     │
│  • Overall confidence: 0.75                                  │
│  • Accept clustering: YES                                    │
│  • Flags: ["cluster_1_metadata_heterogeneous"]               │
│  • Recommendation: "Cluster 1 may benefit from sub-analysis" │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. File Organization (Target Structure)

```
zzy_llm/Tools/APS_XAS/
│
├── 📁 xas_reader/                 # EXISTING: Data loading
│   ├── xas_reader.py              #   - XDI/ASCII parser
│   └── xas_reference_loader.py    #   - Reference spectrum loader
│
├── 📁 xas_analyzer/               # EXISTING + EXTENDED
│   ├── xas_analyzer_main.py       #   - Main preprocessing logic
│   ├── xas_normalization.py       #   - Normalization module
│   ├── xas_models.py              #   ← EXTEND with new models
│   ├── xas_feature_extractor.py   #   ← NEW: Feature extraction
│   └── ...
│
├── 📁 xas_ml_modules/             # NEW: ML analysis suite
│   ├── __init__.py
│   ├── xas_batch_assembler.py     #   - Build XASDataset
│   ├── xas_pca_analyzer.py        #   - PCA implementation
│   ├── xas_clustering.py          #   - Clustering algorithms
│   ├── xas_trend_analyzer.py      #   - Correlation analysis
│   ├── xas_batch_processor.py     #   - Main batch orchestrator
│   ├── xas_ml_plotter.py          #   - ML-specific plots
│   └── README.md                  #   - ML module documentation
│
├── 📁 xas_plotter/                # EXISTING: Visualization
│   ├── xas_quality_report_plots.py
│   └── ...
│
├── 📁 xas_config/                 # EXISTING: Configuration
│   └── xas_plot_settings.yaml
│
├── 📁 tests/                      # NEW: Test suite
│   ├── test_xas_ml/
│   │   ├── test_feature_extractor.py
│   │   ├── test_pca_analyzer.py
│   │   ├── test_clustering.py
│   │   └── test_ml_integration.py
│   └── ...
│
├── 📁 examples/                   # NEW: Examples
│   └── xas_ml_workflow_example.ipynb
│
├── 📄 xas_workflow.py             # EXISTING: Main entry point
├── 📄 xas_automated_processing_workflow_spec.md  # EXISTING: Base spec
├── 📄 xas_ml_integration_spec.md  # NEW: ML spec
├── 📄 xas_ml_roadmap.md           # NEW: Implementation plan
├── 📄 xas_ml_planning_summary.md  # NEW: Summary
└── 📄 xas_architecture_diagram.md # NEW: This file
```

---

## 7. Agent Interaction Pattern

```
┌─────────────────────────────────────────────────────────────┐
│  Agent receives task:                                       │
│  "Analyze 150 Fe K-edge spectra from temperature series"   │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Call Batch Processor                               │
│  processor.process_directory(                               │
│      data_dir="/path/to/Fe_temp_series/",                   │
│      reference="Fe_foil.xdi"                                │
│  )                                                           │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Preprocessing runs for all 150 files               │
│  ✓ 142 valid, 8 excluded (5 invalid, 3 warning)             │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Feature extraction + assembly                      │
│  → XASDataset(n_samples=142, n_features=18)                 │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: ML analysis                                        │
│  • PCA: 5 components explain 95% variance                   │
│  • Clustering: 4 groups identified (silhouette=0.62)        │
│  • Trends: white_line ↔ temperature (r=-0.72, p<0.001)      │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Agent reads structured output                      │
│  {                                                           │
│    "clustering": {"n_clusters": 4, "confidence": 0.75},     │
│    "trend_analysis": {                                      │
│      "significant_correlations": [                          │
│        {"feature": "white_line_intensity",                  │
│         "metadata": "temperature",                          │
│         "r": -0.72, "p_value": 0.0001}                      │
│      ]                                                       │
│    },                                                        │
│    "agent_recommendations": [                               │
│      {"priority": "high",                                   │
│       "message": "Temperature strongly affects oxidation",  │
│       "action": "include_in_model"}                         │
│    ]                                                         │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 6: Agent generates report                             │
│  "Analysis complete. Identified 4 distinct chemical states  │
│   correlated with temperature. Cluster 0 (Fe2+, 300K),      │
│   Cluster 1 (Fe3+, 500K), ... Recommend including temp      │
│   as feature for predictive modeling."                      │
└─────────────────────────────────────────────────────────────┘
```

---

**End of Architecture Diagram**
