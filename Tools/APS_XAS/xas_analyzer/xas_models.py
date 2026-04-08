"""
XAS Data Models

Pydantic models for structured XAS analysis results and metadata.
Provides machine-usable JSON schemas for XAS features and processing parameters.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Optional pandas import for CSV export
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class XASFeatures(BaseModel):
    """
    Extracted XAS features for machine learning and analysis.
    
    Extended feature set (18+ features) for comprehensive XAS characterization.
    All EXAFS features are optional since they require extended k-range.
    """

    # Sample identifier
    sample_name: str = Field(..., description="Sample identifier")

    # =========================================================================
    # EDGE FEATURES (4)
    # =========================================================================
    e0: float = Field(..., description="Edge energy (E0) in eV")
    edge_step: float = Field(..., description="Edge step height (normalized)")
    edge_slope: Optional[float] = Field(None, description="Edge slope (max of first derivative near E0)")
    pre_edge_area: Optional[float] = Field(None, description="Integrated pre-edge intensity")

    # =========================================================================
    # XANES FEATURES (5) - Near-edge structure
    # =========================================================================
    white_line_intensity: Optional[float] = Field(None, description="White line peak intensity above edge")
    white_line_prominence: Optional[float] = Field(None, description="White line peak prominence above baseline")
    white_line_energy: Optional[float] = Field(None, description="White line peak energy in eV")
    white_line_fwhm: Optional[float] = Field(None, description="White line full-width at half-maximum (eV)")
    xanes_area: Optional[float] = Field(None, description="Integrated XANES area (E0 to E0+50 eV)")
    xanes_centroid: Optional[float] = Field(None, description="Spectral centroid of XANES region (eV)")

    # =========================================================================
    # EXAFS FEATURES (3) - Extended fine structure (optional)
    # =========================================================================
    chi_k_rms: Optional[float] = Field(None, description="RMS amplitude of χ(k)")
    ft_peak_r: Optional[float] = Field(None, description="Fourier transform peak position R (Å)")
    ft_peak_amp: Optional[float] = Field(None, description="Fourier transform peak amplitude")
    
    # =========================================================================
    # DERIVATIVE FEATURES (2) - Spectral derivatives
    # =========================================================================
    first_derivative_max: Optional[float] = Field(None, description="Maximum of first derivative (dμ/dE)")
    second_derivative_zero: Optional[float] = Field(None, description="Zero-crossing of second derivative (eV)")
    
    # =========================================================================
    # STATISTICAL FEATURES (4) - Shape descriptors
    # =========================================================================
    spectral_mean: Optional[float] = Field(None, description="Mean normalized μ(E)")
    spectral_variance: Optional[float] = Field(None, description="Variance of normalized spectrum")
    spectral_skewness: Optional[float] = Field(None, description="Skewness (asymmetry) of spectrum")
    spectral_kurtosis: Optional[float] = Field(None, description="Kurtosis (tailedness) of spectrum")
    
    # =========================================================================
    # LEGACY/OPTIONAL (keep for backward compatibility)
    # =========================================================================
    normalized_mu: Optional[List[float]] = Field(None, description="Full normalized μ(E) array (for plotting)")
    ft_area: Optional[float] = Field(None, description="Integrated FT magnitude area")


class ProcessingMetadata(BaseModel):
    """Metadata about XAS data processing parameters."""

    # Beamline and facility information
    facility: Optional[str] = Field(None, description="Facility name (e.g., NSLS-II, APS)")
    beamline: Optional[str] = Field(None, description="Beamline identifier")
    year: Optional[int] = Field(None, description="Measurement year")
    cycle: Optional[int] = Field(None, description="Beamtime cycle")
    pi: Optional[str] = Field(None, description="Principal investigator")
    proposal: Optional[str] = Field(None, description="Proposal ID")
    scan_id: Optional[int] = Field(None, description="Scan identifier")

    # Sample information
    element: Optional[str] = Field(None, description="Element symbol (e.g., Fe, Cu)")
    edge: Optional[str] = Field(None, description="Absorption edge (K, L1, L2, L3)")

    # Energy calibration and ranges
    e0_measured: Optional[float] = Field(None, description="Measured E0 from file header")
    energy_calibration: Optional[str] = Field(None, description="Energy calibration method/notes")

    # k-space and R-space ranges
    k_range: Optional[Dict[str, float]] = Field(None, description="k-space range used (kmin, kmax)")
    r_range: Optional[Dict[str, float]] = Field(None, description="R-space range used (rmin, rmax)")
    windowing: Optional[str] = Field(None, description="Windowing function used")

    # Background subtraction parameters
    rbkg: Optional[float] = Field(None, description="R value for background spline (Å)")
    spline_params: Optional[Dict[str, Any]] = Field(None, description="Spline fitting parameters")

    # FEFF fitting (if applicable)
    feff_paths: Optional[List[Dict[str, Any]]] = Field(None, description="FEFF paths used in fitting")


class XASProcessingParams(BaseModel):
    """Parameters used for XAS data processing."""

    pre_edge: Dict[str, Any] = Field(default_factory=dict, description="Pre-edge subtraction parameters")
    autobk: Dict[str, Any] = Field(default_factory=dict, description="Background removal parameters")
    xftf: Dict[str, Any] = Field(default_factory=dict, description="Fourier transform parameters")


class XASSampleResult(BaseModel):
    """Complete results for a single XAS sample."""

    sample_name: str = Field(..., description="Sample identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Processing timestamp")

    # Core data
    features: XASFeatures = Field(..., description="Extracted XAS features")
    processing_metadata: ProcessingMetadata = Field(..., description="Processing metadata and parameters")
    processing_params: XASProcessingParams = Field(..., description="Detailed processing parameters")

    # File information
    source_file: Optional[str] = Field(None, description="Original data file path")
    file_format: Optional[str] = Field(None, description="File format (XDI, ASCII, etc.)")

    # Optional: raw data arrays (can be large, so optional)
    energy_range: Optional[Dict[str, float]] = Field(None, description="Energy range (min, max) in eV")
    data_points: Optional[int] = Field(None, description="Number of data points")

    class Config:
        json_encoders = {
            np.ndarray: lambda v: v.tolist() if v is not None else None,
            datetime: lambda v: v.isoformat()
        }


class XASBatchResults(BaseModel):
    """Results for a batch of XAS samples."""

    batch_id: str = Field(..., description="Unique batch identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Batch processing timestamp")

    # Batch summary
    total_samples: int = Field(..., description="Total number of samples processed")
    successful_samples: int = Field(..., description="Number of successfully processed samples")
    failed_samples: int = Field(..., description="Number of failed samples")

    # Results
    samples: Dict[str, XASSampleResult] = Field(..., description="Results for each sample")

    # Processing information
    workflow_version: Optional[str] = Field(None, description="XAS workflow version")
    larch_version: Optional[str] = Field(None, description="Larch library version used")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_csv(self, csv_path: str | Path, include_metadata: bool = True) -> Path:
        """
        Export batch features to CSV file.
        
        Parameters
        ----------
        csv_path : str or Path
            Path for the output CSV file
        include_metadata : bool
            Whether to include processing metadata columns
            
        Returns
        -------
        csv_path : Path
            Path to the created CSV file
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for CSV export. Install with: pip install pandas")
            
        csv_path = Path(csv_path)
        
        # Extract features from all samples
        data_rows = []
        
        for sample_name, sample_result in self.samples.items():
            row = {'sample_name': sample_name}
            
            # Add feature data
            features = sample_result.features
            row.update({
                'e0': features.e0,
                'edge_step': features.edge_step,
                'white_line_intensity': features.white_line_intensity,
                'white_line_energy': features.white_line_energy,
                'xanes_area': features.xanes_area,
                'chi_k_rms': features.chi_k_rms,
                'ft_peak_r': features.ft_peak_r,
                'ft_peak_amp': features.ft_peak_amp,
                'ft_area': features.ft_area
            })
            
            # Add metadata if requested
            if include_metadata and sample_result.processing_metadata:
                metadata = sample_result.processing_metadata
                row.update({
                    'facility': metadata.facility,
                    'beamline': metadata.beamline,
                    'year': metadata.year,
                    'cycle': metadata.cycle,
                    'pi': metadata.pi,
                    'proposal': metadata.proposal,
                    'scan_id': metadata.scan_id
                })
                
            data_rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        
        # Reorder columns for better readability
        feature_columns = [
            'sample_name', 'e0', 'edge_step', 'white_line_intensity', 
            'white_line_energy', 'xanes_area', 'chi_k_rms', 
            'ft_peak_r', 'ft_peak_amp', 'ft_area'
        ]
        
        if include_metadata:
            metadata_columns = ['facility', 'beamline', 'year', 'cycle', 'pi', 'proposal', 'scan_id']
            column_order = feature_columns + metadata_columns
        else:
            column_order = feature_columns
            
        # Keep only columns that exist in the data
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]
        
        # Write CSV with header comments
        with open(csv_path, 'w', encoding='utf-8') as f:
            # Write header with feature descriptions and batch info
            f.write(f"# XAS Batch Results Export\n")
            f.write(f"# Batch ID: {self.batch_id}\n")
            f.write(f"# Timestamp: {self.timestamp.isoformat()}\n")
            f.write(f"# Total samples: {self.total_samples}\n")
            f.write(f"# Successful: {self.successful_samples}, Failed: {self.failed_samples}\n")
            f.write("#\n")
            f.write("# Feature descriptions:\n")
            f.write("#   e0: Edge energy (eV)\n")
            f.write("#   edge_step: Edge step height (normalized)\n")
            f.write("#   white_line_intensity: White line peak intensity\n")
            f.write("#   white_line_energy: White line peak energy (eV)\n")
            f.write("#   xanes_area: XANES region area\n")
            f.write("#   chi_k_rms: RMS of chi(k) in EXAFS region\n")
            f.write("#   ft_peak_r: Main peak position in Fourier transform (A)\n")
            f.write("#   ft_peak_amp: Main peak amplitude in Fourier transform\n")
            f.write("#   ft_area: Area under Fourier transform curve\n")
            if include_metadata:
                f.write("#\n")
                f.write("# Metadata columns:\n")
                f.write("#   facility, beamline, year, cycle, pi, proposal, scan_id\n")
            f.write("#\n")
            
            # Write CSV data
            df.to_csv(f, index=False, float_format='%.6f')
        
        return csv_path


def create_sample_result(results_dict: Dict[str, Any],
                        file_info: Optional[Dict[str, Any]] = None) -> XASSampleResult:
    """
    Create a structured XASSampleResult from processing results.

    Parameters
    ----------
    results_dict : dict
        Results from XASProcessor.process_single_spectrum
    file_info : dict, optional
        File information from detect_xas_file_type

    Returns
    -------
    sample_result : XASSampleResult
        Structured result object
    """
    sample_name = results_dict['sample_name']
    features_raw = results_dict.get('features', {})
    processing_params = results_dict.get('processing_params', {})
    processed_data = results_dict.get('processed_data', {})

    # Extract features with proper naming
    features = XASFeatures(
        e0=features_raw.get('e0', 0.0),
        edge_step=features_raw.get('edge_step', 0.0),
        white_line_intensity=features_raw.get('white_line_peak'),
        white_line_energy=features_raw.get('white_line_energy'),
        xanes_area=features_raw.get('xanes_area'),
        normalized_mu=processed_data.get('mu_norm', []).tolist() if processed_data.get('mu_norm') is not None else None,
        chi_k_rms=features_raw.get('chi_k_rms'),
        ft_peak_r=features_raw.get('ft_peak_r'),
        ft_peak_amp=features_raw.get('ft_peak_amp'),
        ft_area=features_raw.get('ft_area')
    )

    # Create processing metadata
    metadata = ProcessingMetadata(
        facility=file_info.get('facility'),
        beamline=file_info.get('beamline'),
        year=file_info.get('year'),
        cycle=file_info.get('cycle'),
        pi=file_info.get('pi'),
        proposal=file_info.get('proposal'),
        scan_id=file_info.get('scan_id'),
        element=file_info.get('element'),
        edge=file_info.get('edge'),
        e0_measured=file_info.get('e0'),
        energy_calibration=file_info.get('energy_calibration'),
        k_range={
            'kmin': processing_params.get('autobk', {}).get('kmin', 0),
            'kmax': processing_params.get('autobk', {}).get('kmax', 14)
        },
        r_range={
            'rmin': processing_params.get('xftf', {}).get('kmin', 2),  # Approximate
            'rmax': processing_params.get('xftf', {}).get('kmax', 12)  # Approximate
        },
        windowing=processing_params.get('xftf', {}).get('window', 'hanning'),
        rbkg=processing_params.get('autobk', {}).get('rbkg', 1.0)
    )

    # Processing parameters
    proc_params = XASProcessingParams(
        pre_edge=processing_params.get('pre_edge', {}),
        autobk=processing_params.get('autobk', {}),
        xftf=processing_params.get('xftf', {})
    )

    # Energy range
    energy = processed_data.get('energy')
    energy_range = None
    data_points = None
    if energy is not None:
        energy_range = {
            'min': float(np.min(energy)),
            'max': float(np.max(energy))
        }
        data_points = len(energy)

    return XASSampleResult(
        sample_name=sample_name,
        features=features,
        processing_metadata=metadata,
        processing_params=proc_params,
        source_file=file_info.get('file_path') if file_info else None,
        file_format=file_info.get('format') if file_info else None,
        energy_range=energy_range,
        data_points=data_points
    )




def save_batch_results_to_json(batch_results: XASBatchResults,
                              output_file: str | Path) -> str:
    """
    Save batch results to JSON file.

    Parameters
    ----------
    batch_results : XASBatchResults
        Batch results to save
    output_file : str or Path
        Output JSON file path

    Returns
    -------
    file_path : str
        Path to saved JSON file
    """
    output_file = Path(output_file)

    # Convert to dict and save
    results_dict = batch_results.dict()
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"Batch results saved to: {output_file}")
    return str(output_file)


def export_batch_features_to_csv(batch_results: XASBatchResults, 
                                csv_path: str | Path,
                                include_metadata: bool = True) -> Path:
    """
    Export XAS batch features to CSV file.
    
    Convenience function for CSV export that can be used independently.
    
    Parameters
    ----------
    batch_results : XASBatchResults
        Batch results to export
    csv_path : str or Path
        Path for the output CSV file
    include_metadata : bool
        Whether to include processing metadata columns
        
    Returns
    -------
    csv_path : Path
        Path to the created CSV file
        
    Example
    -------
    >>> # Export features to CSV
    >>> csv_file = export_batch_features_to_csv(batch_results, 'features.csv')
    >>> print(f"Features exported to {csv_file}")
    """
    return batch_results.to_csv(csv_path, include_metadata)


def load_batch_results_from_json(json_file: str | Path) -> XASBatchResults:
    """
    Load batch results from JSON file.

    Parameters
    ----------
    json_file : str or Path
        Path to JSON file containing batch results

    Returns
    -------
    batch_results : XASBatchResults
        Loaded batch results
    """
    json_file = Path(json_file)

    with open(json_file, 'r') as f:
        data = json.load(f)

    return XASBatchResults(**data)


def create_plots_from_json_results(json_file: str | Path,
                                  output_dir: str | Path = "feature_comparison_plots") -> Dict[str, str]:
    """
    Load batch results from JSON and create comparison plots.

    Convenience function that combines loading JSON results and creating plots.
    Note: This function requires the plotting module to be available.

    Parameters
    ----------
    json_file : str or Path
        Path to JSON file containing batch results
    output_dir : str or Path
        Output directory for plots

    Returns
    -------
    plot_files : dict
        Dictionary mapping plot types to file paths

    Example
    -------
    >>> # After processing batch and saving to JSON
    >>> plots = create_plots_from_json_results('batch_results.json')
    >>> print(f"Created {len(plots)} comparison plots")
    """
    # Load results from JSON
    batch_results = load_batch_results_from_json(json_file)

    # Try to import and use the plotting functionality
    try:
        from ..xas_plotter.xas_features_plotter import create_feature_comparison_plots
        return create_feature_comparison_plots(batch_results, output_dir)
    except ImportError:
        try:
            from xas_plotter.xas_features_plotter import create_feature_comparison_plots
            return create_feature_comparison_plots(batch_results, output_dir)
        except ImportError:
            print("Warning: Plotting functionality not available. Install plotting dependencies.")
            return {}


# =============================================================================
# ML-SPECIFIC DATA MODELS (for batch analysis)
# =============================================================================

class XASDataset(BaseModel):
    """
    Batch-level dataset for machine learning analysis.
    
    Contains feature matrix, metadata, and sample information for
    multiple XAS spectra ready for PCA, clustering, etc.
    """
    
    # Core data
    feature_matrix: Optional[np.ndarray] = Field(None, description="Feature matrix (n_samples × n_features)")
    feature_names: List[str] = Field(default_factory=list, description="Names of features in matrix")
    sample_names: List[str] = Field(..., description="Sample identifiers")
    
    # Metadata (if using pandas)
    metadata_dict: Optional[Dict[str, List[Any]]] = Field(None, description="Metadata as dict of lists")
    
    # Quality information
    quality_flags: Dict[str, List[str]] = Field(default_factory=dict, description="Per-sample quality flags")
    
    # Dataset info
    n_samples: int = Field(..., description="Number of samples")
    n_features: int = Field(..., description="Number of features")
    
    # Processing info
    dataset_id: Optional[str] = Field(None, description="Unique dataset identifier")
    creation_timestamp: datetime = Field(default_factory=datetime.now, description="Dataset creation time")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist() if v is not None else None,
            np.integer: lambda v: int(v),
            np.floating: lambda v: float(v),
            np.bool_: lambda v: bool(v),
            datetime: lambda v: v.isoformat()
        }
    
    def get_metadata_df(self):
        """Get metadata as pandas DataFrame (if pandas available)."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for DataFrame conversion")
        if self.metadata_dict is None:
            return None
        return pd.DataFrame(self.metadata_dict, index=self.sample_names)


class PCAAnalysisResult(BaseModel):
    """Results from Principal Component Analysis."""
    
    # PCA parameters
    n_components: int = Field(..., description="Number of components selected")
    
    # Variance analysis
    explained_variance: List[float] = Field(..., description="Variance explained by each component")
    cumulative_variance: List[float] = Field(..., description="Cumulative variance explained")
    variance_captured: float = Field(..., description="Total variance captured by selected components")
    
    # PCA results
    loadings: Optional[np.ndarray] = Field(None, description="Feature loadings (n_features × n_components)")
    scores: Optional[np.ndarray] = Field(None, description="Sample scores (n_samples × n_components)")
    
    # Feature importance
    feature_importance: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict, 
        description="Top contributing features per component (feature name + loading)"
    )
    
    # Validation metrics
    kaiser_criterion: int = Field(..., description="Components with eigenvalue > 1")
    stability_score: Optional[float] = Field(None, description="Bootstrap stability score (0-1)")
    
    # Quality
    confidence: float = Field(..., description="Overall confidence in PCA results (0-1)")
    flags: List[str] = Field(default_factory=list, description="Warning/error flags")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist() if v is not None else None,
            np.integer: lambda v: int(v),
            np.floating: lambda v: float(v),
            np.bool_: lambda v: bool(v),
            np.integer: lambda v: int(v),
            np.floating: lambda v: float(v),
            np.bool_: lambda v: bool(v)
        }


class ClusteringResult(BaseModel):
    """Results from clustering analysis."""
    
    # Clustering parameters
    method: str = Field(..., description="Clustering method used")
    n_clusters: int = Field(..., description="Number of clusters")
    
    # Cluster assignments
    labels: Optional[np.ndarray] = Field(None, description="Cluster label for each sample")
    
    # Cluster characteristics
    cluster_centers: Optional[np.ndarray] = Field(None, description="Cluster centers (n_clusters × n_features)")
    cluster_info: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-cluster statistics (size, mean spectrum, metadata)"
    )
    
    # Validation metrics
    silhouette_score: float = Field(..., description="Silhouette score (-1 to 1, higher better)")
    davies_bouldin_index: Optional[float] = Field(None, description="Davies-Bouldin index (lower better)")
    calinski_harabasz_score: Optional[float] = Field(None, description="Calinski-Harabasz score (higher better)")
    
    # Physical validation (XAS-specific)
    spectral_similarity_within: List[float] = Field(
        default_factory=list,
        description="Mean spectral similarity within each cluster"
    )
    spectral_separation_between: Optional[float] = Field(
        None,
        description="Mean spectral separation between clusters"
    )
    
    # Quality
    confidence: float = Field(..., description="Overall clustering confidence (0-1)")
    flags: List[str] = Field(default_factory=list, description="Warning/error flags")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist() if v is not None else None
        }


class TrendAnalysisResult(BaseModel):
    """Results from trend and correlation analysis."""
    
    # Feature-metadata correlations
    correlations: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Correlation coefficients {feature: {metadata: r}}"
    )
    p_values: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Statistical p-values {feature: {metadata: p}}"
    )
    
    # Significant findings
    significant_correlations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of significant feature-metadata correlations"
    )
    
    # Cluster-metadata associations (if clustering was performed)
    cluster_metadata_stats: Optional[Dict[str, Any]] = Field(
        None,
        description="Per-cluster metadata distributions"
    )
    
    # Outlier detection
    outlier_indices: List[int] = Field(default_factory=list, description="Indices of outlier samples")
    outlier_scores: Optional[np.ndarray] = Field(None, description="Outlier score for each sample")
    outlier_method: Optional[str] = Field(None, description="Outlier detection method used")
    
    # Quality
    confidence: float = Field(..., description="Overall confidence in trend analysis (0-1)")
    flags: List[str] = Field(default_factory=list, description="Warning/error flags")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist() if v is not None else None,
            np.integer: lambda v: int(v),
            np.floating: lambda v: float(v),
            np.bool_: lambda v: bool(v)
        }


class XASDatasetAnalysis(BaseModel):
    """
    Complete ML analysis results for an XAS dataset.
    
    Top-level container for all ML analysis outputs.
    This is what gets saved to JSON and consumed by agents.
    """
    
    # Dataset info
    dataset_info: Dict[str, Any] = Field(..., description="Dataset summary statistics")
    
    # ML analysis results
    pca_analysis: Optional[PCAAnalysisResult] = Field(None, description="PCA results")
    clustering: Optional[ClusteringResult] = Field(None, description="Clustering results")
    trend_analysis: Optional[TrendAnalysisResult] = Field(None, description="Trend analysis results")
    
    # Validation summary
    validation_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Overall quality assessment and recommendations"
    )
    
    # Agent recommendations
    agent_recommendations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Actionable recommendations for agents"
    )
    
    # Processing metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    workflow_version: Optional[str] = Field(None, description="ML workflow version")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
