# XAS Automated Pipeline — Implementation Specification

## Overview

Build an automated XAS (X-ray Absorption Spectroscopy) data processing pipeline that takes raw XAS data through preprocessing, feature extraction, ML analysis, and LCF fitting. Each processing stage saves both CSV data and plots in a self-contained subfolder.

---

## Project Structure


project_root/ ├── config/ │ ├── pipeline_config.yaml │ ├── ml_config.yaml │ └── standards_library/ │ ├── Fe2O3.dat │ ├── FeO.dat │ └── Fe_foil.dat │ ├── data/ │ ├── raw/ # input: raw XAS files │ │ │ ├── preprocessed/ # output: step 1 │ │ ├── sample_001_norm.csv │ │ ├── all_normalized_spectra.csv │ │ └── plots/ │ │ ├── sample_001_normalization.png │ │ └── all_spectra_overlay.png │ │ │ ├── features/ # output: step 2 │ │ ├── sample_001_features.csv │ │ ├── features_summary.csv │ │ └── plots/ │ │ ├── edge_position_trend.png │ │ ├── white_line_vs_edge.png │ │ ├── feature_correlation_matrix.png │ │ └── feature_distributions.png │ │ │ ├── ml_results/ # output: step 3 │ │ ├── pca_scores.csv │ │ ├── pca_loadings.csv │ │ ├── pca_variance_explained.csv │ │ ├── cluster_labels.csv │ │ ├── cluster_centers.csv │ │ └── plots/ │ │ ├── pca_scree_plot.png │ │ ├── pca_score_plot_PC1_PC2.png │ │ ├── pca_loadings_plot.png │ │ ├── cluster_scatter.png │ │ └── cluster_spectra_overlay.png │ │ │ ├── lcf_results/ # output: step 4 │ │ ├── sample_001_lcf_fit.csv │ │ ├── sample_001_lcf_components.csv │ │ ├── lcf_summary.csv │ │ └── plots/ │ │ ├── sample_001_lcf_fit.png │ │ ├── lcf_fractions_barplot.png │ │ ├── lcf_fractions_stacked.png │ │ └── lcf_rfactor_summary.png │ │ │ └── reports/ # output: step 5 │ ├── quality_report.csv │ ├── pipeline_summary.csv │ └── plots/ │ └── qc_dashboard.png │ ├── logs/ │ └── pipeline_run.log │ └── src/ ├── pipeline.py ├── preprocessing.py ├── feature_extraction.py ├── ml_analysis.py ├── lcf_fitting.py ├── quality_checks.py └── utils.py


---

## Pipeline Flow


Raw .xdi/.dat files │ ▼ ┌──────────────────┐ │ 1. Preprocessing │ energy calibration, normalization │ (per-sample) │ └────────┬─────────┘ │ normalized μ(E) │ ┌────┴────────────────┐ │ │ ▼ ▼ ┌──────────────┐ ┌──────────────────┐ │ 2. Feature │ │ 3. ML Analysis │ │ Extraction │ │ (PCA on spectra)│ │ (per-sample) │ │ (batch) │ └──────┬───────┘ └────────┬─────────┘ │ │ │ feature table │ PCA scores │ │ │ │ ▼ │ │ ┌─────────────┐ │ │ │ 3. ML cont'd│ │ │ │ (PCA on feat)│ │ │ │ (clustering) │ │ │ └──────┬──────┘ │ │ │ │ └────┬────┘──────────┘ │ │ cluster labels can guide standard selection ▼ ┌──────────────┐ │ 4. LCF │ │ Fitting │ │ (per-sample) │ └──────┬───────┘ │ ▼ ┌──────────────┐ │ 5. Reporting │ │ & QC │ └──────────────┘


---

## Configuration Files

### `config/pipeline_config.yaml`

```yaml
pipeline:
  # Input
  raw_file_pattern: "*.xdi"  # also support *.dat, *.csv

  # Preprocessing
  preprocessing:
    energy_column: "energy"
    mu_column: "mu"
    e0_method: "max_deriv"          # or "half_edge", "fixed"
    e0_fixed: null                  # used if e0_method is "fixed"
    energy_shift: 0.0               # manual energy correction in eV
    pre_edge_range: [-150, -30]     # relative to E0
    post_edge_range: [50, 300]      # relative to E0
    normalization_order: 2          # polynomial order for post-edge

  # Feature Extraction
  feature_extraction:
    features:
      - edge_position
      - white_line_intensity
      - white_line_position
      - pre_edge_area
      - pre_edge_centroid
      - edge_width
      - post_edge_slope
    reference_e0: null              # if set, compute edge_shift_from_ref

  # LCF
  lcf:
    standards_dir: "config/standards_library"
    fit_range: [-20, 30]            # relative to E0
    force_positive: true
    sum_to_one: true
    combinatorial: true
    max_components: 3
    min_fraction: 0.01              # drop components below this

  # Quality Checks
  quality:
    min_edge_step: 0.1
    max_r_factor: 0.02
    max_chi_sq: 0.05

config/ml_config.yaml
ml:
  pca:
    n_components: 5
    input: "spectra"                # "spectra", "features", or "both"
    scale: true                     # StandardScaler before PCA

  clustering:
    method: "kmeans"                # "kmeans", "dbscan", "agglomerative"
    input: "pca_scores"             # "pca_scores", "features", "spectra"
    n_clusters: 3                   # for kmeans/agglomerative
    eps: 0.5                        # for dbscan
    min_samples: 3                  # for dbscan

  anomaly_detection:
    enabled: true
    method: "isolation_forest"      # or "pca_residual"
    contamination: 0.1

Module Specifications
1. src/utils.py — Shared Utilities
"""
Shared utility functions used across all modules.
"""

# Functions to implement:

def load_config(config_path: str) -> dict:
    """Load YAML config file and return dict."""

def setup_logging(log_dir: Path, run_name: str) -> logging.Logger:
    """Configure logging to file and console."""

def ensure_dir(path: Path) -> Path:
    """Create directory (and plots/ subdir) if it doesn't exist. Return path."""

def load_xas_file(filepath: Path) -> pd.DataFrame:
    """
    Load raw XAS data file (.xdi, .dat, .csv).
    Return DataFrame with at minimum columns: 'energy', 'mu'.
    Handle comment lines (# or %) and varying delimiters.
    """

def interpolate_to_grid(energy, mu, grid) -> np.ndarray:
    """Interpolate spectrum onto a common energy grid."""

def build_spectra_matrix(spectra: dict) -> tuple:
    """
    Given dict of {sample_id: {"energy": array, "mu_norm": array}},
    interpolate all onto common grid.
    
    Returns:
        spectra_matrix: np.ndarray, shape (n_samples, n_grid_points)
        sample_ids: list of str
        energy_grid: np.ndarray
    """

def build_feature_matrix(features: dict) -> tuple:
    """
    Given dict of {sample_id: XASFeatures dataclass},
    build a numeric matrix.
    
    Returns:
        feature_matrix: np.ndarray, shape (n_samples, n_features)
        sample_ids: list of str
        feature_names: list of str
    """

2. src/preprocessing.py — Preprocessing Module
"""
XAS Preprocessing: energy calibration, normalization, quality checks.

Depends on: utils.py
Outputs to: data/preprocessed/
"""

class XASPreprocessor:
    def __init__(self, config: dict):
        """
        Store preprocessing config parameters:
        - pre_edge_range, post_edge_range
        - e0_method, energy_shift
        - normalization_order
        """

    def process(self, filepath: Path) -> dict:
        """
        Full preprocessing pipeline for a single sample.
        
        Steps:
            1. Load raw data using utils.load_xas_file()
            2. Apply energy shift if configured
            3. Find E0 (edge energy) using configured method
            4. Fit pre-edge line (linear fit in pre_edge_range)
            5. Fit post-edge line/polynomial (in post_edge_range)
            6. Normalize: mu_norm = (mu - pre_edge) / edge_step
        
        Returns dict:
            {
                "sample_id": str,
                "energy": np.ndarray,
                "mu_raw": np.ndarray,
                "mu_norm": np.ndarray,
                "e0": float,
                "edge_step": float,
                "pre_edge_line": np.ndarray,
                "post_edge_line": np.ndarray,
            }
        """

    def find_e0(self, energy: np.ndarray, mu: np.ndarray) -> float:
        """
        Find edge energy E0.
        Methods:
            - "max_deriv": energy at maximum of first derivative
            - "half_edge": energy at half the edge step
            - "fixed": use config value
        """

    def plot_normalization(self, norm_data: dict, save_path: Path):
        """
        Plot showing:
            - Raw mu(E)
            - Pre-edge and post-edge fit lines
            - Normalized mu(E) (on secondary axis or separate panel)
            - E0 marked with vertical line
        Save to save_path.
        """

    def plot_overlay(self, all_spectra: dict, save_path: Path):
        """
        Overlay all normalized spectra on one plot.
        X-axis: energy (eV), Y-axis: normalized mu.
        Include legend with sample IDs.
        Save to save_path.
        """

    def save_csv(self, norm_data: dict, out_dir: Path):
        """
        Save per-sample CSV with columns:
            energy, mu_raw, mu_norm, pre_edge_line, post_edge_line
        """

    def save_combined_csv(self, all_spectra: dict, out_dir: Path):
        """
        Save all_normalized_spectra.csv.
        Columns: energy, sample_001, sample_002, ...
        All spectra interpolated onto common energy grid.
        """

3. src/feature_extraction.py — Feature Extraction Module
"""
Extract scalar spectral features from normalized XAS data.

Depends on: preprocessing output
Outputs to: data/features/
"""

from dataclasses import dataclass, asdict

@dataclass
class XASFeatures:
    sample_id: str
    edge_position: float          # E0 in eV
    white_line_intensity: float   # max of normalized mu near edge
    white_line_position: float    # energy at white line max (eV)
    pre_edge_area: float          # integrated pre-edge peak area
    pre_edge_centroid: float      # energy centroid of pre-edge
    edge_width: float             # 20%-80% rise width in eV
    post_edge_slope: float        # linear slope of post-edge
    edge_shift_from_ref: float    # shift relative to reference E0


class XASFeatureExtractor:
    def __init__(self, config: dict):
        """Store feature extraction config."""

    def extract(self, norm_data: dict) -> XASFeatures:
        """
        Extract all configured features from one normalized spectrum.
        
        Parameters:
            norm_data: dict from XASPreprocessor.process()
        
        Returns:
            XASFeatures dataclass
        
        Feature definitions:
            - edge_position: E0 from preprocessing
            - white_line_intensity: max(mu_norm) in range [E0, E0+20]
            - white_line_position: energy at that max
            - pre_edge_area: np.trapz(mu_norm) in range [E0-30, E0-5]
            - pre_edge_centroid: weighted average energy in pre-edge range
            - edge_width: energy difference between 20% and 80% of edge rise
            - post_edge_slope: linear fit slope in [E0+50, E0+150]
            - edge_shift_from_ref: E0 - reference_e0 (if configured)
        """

    def extract_all(self, spectra: dict) -> dict:
        """
        Extract features for all samples.
        Returns dict of {sample_id: XASFeatures}.
        """

    def save_csv(self, features: dict, out_dir: Path):
        """
        Save:
            - Per-sample: {sample_id}_features.csv (one row)
            - Summary: features_summary.csv (all samples, one row each)
        """

    def plot_trends(self, summary_df: pd.DataFrame, plot_dir: Path):
        """
        Generate and save:
            1. edge_position_trend.png
               - X: sample index or sample_id, Y: edge position
            2. white_line_vs_edge.png
               - scatter: X=edge_position, Y=white_line_intensity
            3. feature_correlation_matrix.png
               - heatmap of pairwise correlations between all features
            4. feature_distributions.png
               - histogram/violin for each feature
        """

4. src/ml_analysis.py — ML Analysis Module
"""
ML analysis: PCA, clustering, anomaly detection.
Operates on spectra matrices and/or feature matrices.

Depends on: preprocessing output, feature extraction output
Outputs to: data/ml_results/
"""

class PCAAnalyzer:
    def __init__(self, config: dict):
        """
        Config:
            - n_components: int
            - scale: bool (StandardScaler before PCA)
        """

    def fit_transform(self, matrix: np.ndarray, sample_ids: list,
                      column_labels=None) -> dict:
        """
        Run PCA on matrix (n_samples, n_features).
        
        If scale=True, apply StandardScaler first.
        
        Returns dict:
            {
                "scores": np.ndarray (n_samples, n_components),
                "loadings": np.ndarray (n_components, n_features),
                "variance_explained": np.ndarray (n_components,),
                "cumulative_variance": np.ndarray (n_components,),
                "sample_ids": list,
                "column_labels": list or np.ndarray,
            }
        """


class Clusterer:
    def __init__(self, config: dict):
        """
        Config:
            - method: "kmeans" | "dbscan" | "agglomerative"
            - n_clusters, eps, min_samples as appropriate
        """

    def fit(self, data: np.ndarray, sample_ids: list) -> dict:
        """
        Cluster the data.
        
        Returns dict:
            {
                "labels": np.ndarray (n_samples,),
                "centers": np.ndarray (n_clusters, n_features) or None,
                "sample_ids": list,
                "n_clusters": int,
                "method": str,
            }
        """


class AnomalyDetector:
    def __init__(self, config: dict):
        """Config: method, contamination."""

    def detect(self, data: np.ndarray, sample_ids: list) -> dict:
        """
        Returns dict:
            {
                "is_anomaly": np.ndarray of bool,
                "scores": np.ndarray of float,
                "sample_ids": list,
            }
        """


class MLAnalyzer:
    def __init__(self, config: dict):
        """Initialize PCAAnalyzer, Clusterer, AnomalyDetector from config."""

    def run(self, spectra_matrix: np.ndarray = None,
            feature_matrix: np.ndarray = None,
            sample_ids: list = None,
            energy_grid: np.ndarray = None,
            feature_names: list = None) -> dict:
        """
        Run all configured ML analyses.
        
        Logic:
            1. Determine PCA input based on config (spectra, features, or both)
            2. Run PCA
            3. Determine clustering input (pca_scores, features, or spectra)
            4. Run clustering
            5. Run anomaly detection if enabled
        
        Returns dict:
            {
                "pca": {PCA results dict},
                "pca_features": {PCA results dict} or None,  # if input="both"
                "clustering": {clustering results dict},
                "anomalies": {anomaly results dict} or None,
            }
        """

    def save_csv(self, ml_results: dict, sample_ids: list, out_dir: Path):
        """
        Save:
            - pca_scores.csv: columns = [sample_id, PC1, PC2, ...]
            - pca_loadings.csv: rows = PCs, columns = energy/feature labels
            - pca_variance_explained.csv: columns = [PC, variance, cumulative]
            - cluster_labels.csv: columns = [sample_id, cluster]
            - cluster_centers.csv
        """

    def plot_all(self, ml_results: dict, spectra: dict, plot_dir: Path):
        """
        Generate and save:
            1. pca_scree_plot.png
               - bar chart of variance explained per PC
               - line overlay of cumulative variance
            2. pca_score_plot_PC1_PC2.png
               - scatter of PC1 vs PC2
               - color by cluster label if available
               - annotate with sample_ids
            3. pca_loadings_plot.png
               - line plot of first 3 PC loadings vs energy
            4. cluster_scatter.png
               - scatter of PC1 vs PC2 colored by cluster
            5. cluster_spectra_overlay.png
               - one subplot per cluster
               - overlay all spectra in that cluster
               - plot cluster center spectrum
        """

5. src/lcf_fitting.py — LCF Fitting Module
"""
Linear Combination Fitting of XAS spectra using reference standards.

Depends on: preprocessing output, optionally ML results for standard selection
Outputs to: data/lcf_results/
"""

class LCFFitter:
    def __init__(self, config: dict):
        """
        Load config and reference standards from standards_dir.
        
        Store:
            - standards: dict of {name: {"energy": array, "mu": array}}
            - fit_range, force_positive, sum_to_one, etc.
        """

    def load_standards(self, standards_dir: Path) -> dict:
        """
        Load all standard spectra from directory.
        Each file has columns: energy, mu (or mu_norm).
        Return dict of {filename_stem: {"energy": array, "mu": array}}.
        """

    def fit(self, norm_data: dict, standards: list = None) -> dict:
        """
        Perform LCF on one sample.
        
        Steps:
            1. Select standards to use (all, or subset from `standards` arg)
            2. Interpolate sample and standards onto common energy grid
               within fit_range relative to E0
            3. If combinatorial=True, try all combinations of 1..max_components
            4. For each combination, solve least-squares:
               mu_sample = sum(f_i * mu_standard_i)
               with constraints: f_i >= 0 (if force_positive), sum(f_i) = 1 (if sum_to_one)
            5. Select best fit by R-factor
        
        Use scipy.optimize.minimize or scipy.optimize.nnls.
        
        R-factor = sum((data - fit)^2) / sum(data^2)
        
        Returns dict:
            {
                "sample_id": str,
                "energy": np.ndarray (fit range),
                "mu_data": np.ndarray,
                "mu_fit": np.ndarray,
                "residual": np.ndarray,
                "fractions": {"Fe2O3": 0.6, "FeO": 0.35, "Fe_foil": 0.05},
                "components": {"Fe2O3": array, "FeO": array, "Fe_foil": array},
                "r_factor": float,
                "chi_sq": float,
                "standards_used": list of str,
                "all_combinations": list of dicts (optional, for reporting),
            }
        """

    def fit_all(self, spectra: dict, ml_results: dict = None) -> dict:
        """
        Fit all samples. Optionally use ML cluster labels to select standards.
        Returns dict of {sample_id: {"result": lcf_dict, "qc": qc_dict}}.
        """

    def save_csv(self, lcf_results: dict, out_dir: Path):
        """
        Save:
            - Per-sample: {sample_id}_lcf_fit.csv
              columns: energy, mu_data, mu_fit, residual
            - Per-sample: {sample_id}_lcf_components.csv
              columns: energy, Fe2O3, FeO, Fe_foil (weighted contributions)
            - Summary: lcf_summary.csv
              columns: sample_id, Fe2O3_frac, FeO_frac, ..., r_factor, chi_sq, qc_passed
        """

    def plot_fit(self, lcf_result: dict, save_path: Path):
        """
        Single sample LCF fit plot:
            - Data points (or line)
            - Fit line
            - Individual weighted standard contributions (stacked or overlaid)
            - Residual (below, separate panel)
            - Title with sample_id, R-factor
            - Legend with fractions
        """

    def plot_fractions_bar(self, lcf_results: dict, save_path: Path):
        """
        Grouped bar chart:
            X-axis: sample_id
            Y-axis: fraction
            One bar group per sample, one color per standard.
        """

    def plot_fractions_stacked(self, lcf_results: dict, save_path: Path):
        """
        Stacked bar chart:
            X-axis: sample_id
            Y-axis: fraction (0 to 1)
            Stacked colors per standard.
        """

    def plot_rfactor_summary(self, lcf_results: dict, save_path: Path):
        """
        Bar chart of R-factor per sample.
        Horizontal line at quality threshold.
        Color bars red if above threshold.
        """

6. src/quality_checks.py — Quality Checks
"""
Quality checks for each pipeline stage.

Depends on: config thresholds
"""

class QualityChecker:
    def __init__(self, config: dict):
        """Store quality thresholds from config."""

    def check_preprocessing(self, norm_data: dict) -> dict:
        """
        Check:
            - edge_step > min_edge_step
            - E0 is within expected range
            - no NaN/inf in normalized data
            - normalized mu starts near 0 and ends near 1
        
        Returns:
            {"passed": bool, "reason": str or None, "details": dict}
        """

    def check_lcf(self, lcf_result: dict) -> dict:
        """
        Check:
            - r_factor < max_r_factor
            - chi_sq < max_chi_sq
            - all fractions are physically reasonable
            - fractions sum to ~1.0 (within tolerance)
        
        Returns:
            {"passed": bool, "reason": str or None, "details": dict}
        """

    def plot_dashboard(self, spectra, features, ml_results, lcf_results,
                       save_path: Path):
        """
        Multi-panel QC dashboard:
            Panel 1: Edge step distribution (histogram)
            Panel 2: R-factor per sample (bar, with threshold line)
            Panel 3: PCA score plot colored by QC pass/fail
            Panel 4: Summary statistics table
        """

7. src/pipeline.py — Pipeline Orchestrator
"""
Main pipeline orchestrator. Coordinates all modules.

Usage:
    from pipeline import XASPipeline
    pipe = XASPipeline("config/pipeline_config.yaml")
    pipe.run(Path("project_root"))
"""

class XASPipeline:
    def __init__(self, config_path: str):
        """
        Load config files.
        Initialize all modules:
            - XASPreprocessor
            - XASFeatureExtractor
            - MLAnalyzer
            - LCFFitter
            - QualityChecker
        Setup logging.
        """

    def run(self, project_root: Path):
        """
        Execute full pipeline.
        
        Steps:
            1. Create output directory structure
            2. Run preprocessing
            3. Run feature extraction
            4. Run ML analysis
            5. Run LCF fitting
            6. Run quality checks and reporting
        
        Each step:
            a. Process data
            b. Save CSVs to appropriate subfolder
            c. Save plots to subfolder/plots/
            d. Log progress and any warnings
        """

    def _setup_directories(self, project_root: Path) -> dict:
        """
        Create all output directories:
            data/preprocessed/plots/
            data/features/plots/
            data/ml_results/plots/
            data/lcf_results/plots/
            data/reports/plots/
            logs/
        
        Returns dict mapping stage names to Path objects.
        """

    def _run_preprocessing(self, raw_dir: Path, out_dir: Path) -> dict:
        """
        For each raw file:
            1. preprocessor.process(file)
            2. quality check
            3. save CSV
            4. save normalization plot
        Then:
            5. save combined spectra CSV
            6. save overlay plot
        
        Returns: dict of {sample_id: norm_data}
        """

    def _run_feature_extraction(self, spectra: dict, out_dir: Path) -> dict:
        """
        1. Extract features for all samples
        2. Save per-sample CSVs
        3. Save features_summary.csv
        4. Generate and save feature plots
        
        Returns: dict of {sample_id: XASFeatures}
        """

    def _run_ml_analysis(self, spectra: dict, features: dict,
                         out_dir: Path) -> dict:
        """
        1. Build spectra matrix and feature matrix (using utils)
        2. Run ML analyzer
        3. Save all CSVs (PCA, clustering)
        4. Generate and save all ML plots
        
        Returns: ml_results dict
        """

    def _run_lcf(self, spectra: dict, ml_results: dict,
                 out_dir: Path) -> dict:
        """
        For each sample:
            1. Optionally get suggested standards from ML clusters
            2. Run LCF fit
            3. Quality check
            4. Save fit CSV and components CSV
            5. Save fit plot
        Then:
            6. Save lcf_summary.csv
            7. Save summary plots (bar, stacked, R-factor)
        
        Returns: dict of {sample_id: {"result": dict, "qc": dict}}
        """

    def _run_reporting(self, spectra, features, ml_results,
                       lcf_results, out_dir: Path):
        """
        1. Build quality_report.csv (per-sample QC summary)
        2. Build pipeline_summary.csv (aggregate metrics)
        3. Generate QC dashboard plot
        """

Implementation Notes
Dependencies
numpy
pandas
scipy
scikit-learn
matplotlib
pyyaml

Key Conventions
Every module follows the pattern: process() → save_csv() → plot()
Every output subfolder contains both .csv files and a plots/ subdirectory
All plots: use matplotlib, call plt.savefig(path, dpi=150, bbox_inches='tight'), then plt.close()
All CSVs: use pandas.DataFrame.to_csv(path, index=False) unless index is meaningful (like sample_id)
Logging: use Python logging module, log to both console and logs/pipeline_run.log
Error handling: wrap per-sample processing in try/except, log errors, continue to next sample
NaN handling: use np.nan for features that cannot be computed; flag in QC
Data Format Assumptions
Raw files: two-column (energy, mu) with optional header lines starting with #
Energy in eV
Standards files: same format as raw files, already normalized
All spectra at the same absorption edge (e.g., Fe K-edge)
Plotting Style
Use consistent color scheme across all plots
Include axis labels with units
Include titles
Use plt.style.use('seaborn-v0_8-whitegrid') or similar clean style
For per-sample plots: include sample_id in title
For summary plots: include number of samples in title
Entry Point
# run_pipeline.py
from pathlib import Path
from src.pipeline import XASPipeline

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="XAS Automated Pipeline")
    parser.add_argument("project_root", type=str, help="Path to project root")
    parser.add_argument("--config", type=str, default="config/pipeline_config.yaml")
    args = parser.parse_args()

    pipeline = XASPipeline(args.config)
    pipeline.run(Path(args.project_root))

Testing

Create test data in data/raw/ with at least 5 synthetic XAS spectra:

Use an arctangent function for the edge
Add Gaussian pre-edge peak
Add Gaussian white line
Vary E0, white line intensity, and pre-edge area across samples
Add small random noise

This allows end-to-end testing without real beamline data. ```