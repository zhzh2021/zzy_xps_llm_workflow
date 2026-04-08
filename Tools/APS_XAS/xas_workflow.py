"""
XAS Automated Processing Workflow

Production-grade, agent-safe, modular workflow for XAS data processing.
Executes steps in canonical order and emits structured results for agent consumption.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Import modular components
try:
    from .xas_reader.xas_reader import read_xas_file
    from .xas_reader.xas_reference_loader import load_xas_reference
    from .xas_analyzer.xas_analyzer_main import XASProcessor
    from .xas_analyzer.spectrum_quality_check import XASQualityReportGenerator
    from .xas_plotter.xas_quality_report_plots import generate_xas_quality_report
    from .xas_plotter.xas_plotter_main import XASPlotter
    from .xas_plotter.quality_control_plotter import create_xas_diagnostic_plot
except ImportError:
    from xas_reader.xas_reader import read_xas_file
    from xas_reader.xas_reference_loader import load_xas_reference
    from xas_analyzer.xas_analyzer_main import XASProcessor
    from xas_analyzer.spectrum_quality_check import XASQualityReportGenerator
    from xas_plotter.xas_quality_report_plots import generate_xas_quality_report
    from xas_plotter.quality_control_plotter import create_xas_diagnostic_plot
    from xas_plotter.xas_plotter_main import XASPlotter

# Import ML modules (optional - only if ML analysis is enabled)
try:
    from .xas_analyzer.xas_feature_extractor import XASFeatureExtractor
    from .xas_ml_modules.xas_batch_assembler import XASBatchAssembler
    from .xas_ml_modules.xas_pca_analyzer import XASPCAAnalyzer
    from .xas_ml_modules.xas_clusterer import XASClusterer
    from .xas_ml_modules.xas_trend_analyzer import XASTrendAnalyzer
    from .xas_analyzer.xas_models import XASFeatures, ProcessingMetadata, XASProcessingParams
    HAS_ML_MODULES = True
except ImportError:
    try:
        from xas_analyzer.xas_feature_extractor import XASFeatureExtractor
        from xas_ml_modules.xas_batch_assembler import XASBatchAssembler
        from xas_ml_modules.xas_pca_analyzer import XASPCAAnalyzer
        from xas_ml_modules.xas_clusterer import XASClusterer
        from xas_ml_modules.xas_trend_analyzer import XASTrendAnalyzer
        from xas_analyzer.xas_models import XASFeatures, ProcessingMetadata, XASProcessingParams
        HAS_ML_MODULES = True
    except ImportError:
        HAS_ML_MODULES = False


class XASAutomatedProcessor:
    """
    Automated XAS processing workflow following canonical order.

    Executes: raw data (reader) → reference(reader) → energy alignment (analyzer) → deglitching (analyzer) → normalization (analyzer) → validation (analyzer) → spectrum QC (analyzer) → plotting (plotter)
    Emits structured results for agent consumption.
    """

    def __init__(self,
                 output_dir: Optional[str | Path] = None,
                 reference_file: Optional[str] = None,
                 create_diagnostic_plots: bool = False,
                 enable_ml_analysis: bool = False):
        """
        Initialize automated processor.

        Parameters
        ----------
        output_dir : str or Path, optional
            Output directory for results
        reference_file : str, optional
            Path to reference spectrum file
        create_diagnostic_plots : bool
            Whether to generate diagnostic plots
        enable_ml_analysis : bool
            Whether to run ML analysis on batch data (requires ML modules)
        """
        # Default output directory
        if output_dir is None:
            project_root = Path(__file__).resolve().parents[2] / "project_root"
            self.output_dir = project_root / "xas_results"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate plots directory
        project_root = Path(__file__).resolve().parents[2] / "project_root"
        self.plots_dir = project_root / "xas_plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.reference_file = reference_file
        self.create_diagnostic_plots = create_diagnostic_plots
        self.enable_ml_analysis = enable_ml_analysis and HAS_ML_MODULES
        
        if enable_ml_analysis and not HAS_ML_MODULES:
            print("Warning: ML analysis requested but ML modules not available. Skipping ML analysis.")
        
        # Initialize ML modules if enabled
        if self.enable_ml_analysis:
            self.feature_extractor = XASFeatureExtractor()
            self.batch_assembler = XASBatchAssembler()
            self.pca_analyzer = XASPCAAnalyzer()
            self.clusterer = XASClusterer()
            self.trend_analyzer = XASTrendAnalyzer()

        # Initialize analyzer with reference data
        reference_energy = None
        reference_mu = None
        if reference_file:
            ref_result = load_xas_reference(reference_file)
            if ref_result["reference_energy"] is not None:
                reference_energy = ref_result["reference_energy"]
                reference_mu = ref_result["reference_mu"]

        alignment_method = 'reference' if reference_energy is not None else 'derivative'
        self.analyzer = XASProcessor(
            alignment_method=alignment_method,
            reference_energy=reference_energy,
            reference_mu=reference_mu
        )
        
        # Initialize plotter with correct settings file path
        settings_file = Path(__file__).parent / "xas_config" / "xas_plot_settings.yaml"
        self.plotter = XASPlotter(settings_file=settings_file)
    
    def _clean_arrays_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively remove numpy arrays and lists from nested dictionary.
        Keep only metadata (scalars, strings, small numbers).
        
        Parameters
        ----------
        data : dict
            Dictionary to clean
            
        Returns
        -------
        cleaned : dict
            Dictionary with arrays removed
        """
        cleaned = {}
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively clean nested dictionaries
                cleaned[key] = self._clean_arrays_from_dict(value)
            elif isinstance(value, (np.ndarray, list)):
                # Skip arrays and lists (these are data, not metadata)
                if isinstance(value, list) and len(value) > 10:
                    # Skip long lists (likely data arrays)
                    continue
                elif isinstance(value, np.ndarray):
                    # Skip all numpy arrays
                    continue
                else:
                    # Keep short lists (flags, etc.)
                    cleaned[key] = value
            else:
                # Keep scalars, strings, booleans, None
                cleaned[key] = value
        return cleaned

    def process_single_spectrum(self,
                               file_path: str | Path,
                               sample_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process single XAS spectrum through complete workflow.

        Parameters
        ----------
        file_path : str or Path
            Path to XAS data file
        sample_name : str, optional
            Sample identifier

        Returns
        -------
        result : dict
            Structured processing results
        """
        file_path = Path(file_path)
        if sample_name is None:
            sample_name = file_path.stem

        # Initialize result structure (metadata only - no data arrays)
        result = {
            "sample_name": sample_name,
            "file_path": str(file_path),
            "processing_timestamp": datetime.now().isoformat(),
            "metadata": {},
            "reference": {},
            "energy_alignment": {},
            "deglitching": {},
            "normalization": {},
            "normalization_validation": {},
            "spectrum_quality": {},
            "output_files": {},
            "success": False,
            "error": None
        }

        try:
            # Step 1: Raw Data Loader
            print(f"Loading raw data: {sample_name}")
            raw_data = read_xas_file(file_path)
            result["metadata"] = raw_data["metadata"]

            energy = raw_data["energy"]
            mu = raw_data["mu"]
            
            # Save raw data to CSV
            csv_dir = self.output_dir / "processed_data"
            csv_dir.mkdir(exist_ok=True)
            raw_csv_path = csv_dir / f"{sample_name}_raw.csv"
            np.savetxt(raw_csv_path, np.column_stack([energy, mu]),
                      delimiter=",", header="energy_eV,mu", comments="")
            result["output_files"]["raw_csv"] = str(raw_csv_path)

            # Step 2: Reference Loader (optional)
            if self.reference_file:
                ref_result = load_xas_reference(self.reference_file)
                result["reference"] = ref_result
            else:
                result["reference"] = {
                    "reference_energy": None,
                    "reference_mu": None,
                    "flags": ["no_reference_provided"]
                }

            # Step 3-7: Analyzer processing (energy alignment → deglitching → normalization → validation → quality assessment)
            print(f"Processing spectrum: {sample_name}")
            analysis_result = self.analyzer.process_single_spectrum(
                raw_data["energy"], raw_data["mu"], sample_name
            )

            # Extract processed data arrays before merging
            processed_data = analysis_result.pop("processed_data", {})
            
            # Remove array data from nested results (keep only metadata/scalars)
            analysis_result = self._clean_arrays_from_dict(analysis_result)
            
            # Merge analyzer results (metadata only) into main result
            result.update(analysis_result)
            
            # Save processed data to CSV files
            if processed_data:
                # Save cleaned data
                if "mu_cleaned" in processed_data and processed_data["mu_cleaned"] is not None:
                    cleaned_csv_path = csv_dir / f"{sample_name}_cleaned.csv"
                    np.savetxt(cleaned_csv_path, 
                              np.column_stack([processed_data["energy"], processed_data["mu_cleaned"]]),
                              delimiter=",", header="energy_eV,mu_cleaned", comments="")
                    result["output_files"]["cleaned_csv"] = str(cleaned_csv_path)
                
                # Save normalized data
                if "mu_normalized" in processed_data and processed_data["mu_normalized"] is not None:
                    normalized_csv_path = csv_dir / f"{sample_name}_normalized.csv"
                    np.savetxt(normalized_csv_path,
                              np.column_stack([processed_data["energy"], processed_data["mu_normalized"]]),
                              delimiter=",", header="energy_eV,mu_normalized", comments="")
                    result["output_files"]["normalized_csv"] = str(normalized_csv_path)

            # Step 8: Individual Plot Generation
            if self.create_diagnostic_plots and processed_data:
                print(f"Creating individual plots: {sample_name}")
                # Create sample-specific subfolder
                sample_plot_dir = self.plots_dir / sample_name
                sample_plot_dir.mkdir(parents=True, exist_ok=True)
                
                # Create diagnostic plot
                if "mu_cleaned" in processed_data and "mu_normalized" in processed_data:
                    fig = create_xas_diagnostic_plot(
                        processed_data["energy"], mu, 
                        processed_data["mu_cleaned"], processed_data["mu_normalized"],
                        result.get("normalization", {}).get("parameters", {}), 
                        sample_name, str(sample_plot_dir)
                    )
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    diagnostic_plot_path = sample_plot_dir / f"{sample_name}_diagnostic.png"
                    result["output_files"]["diagnostic_plot"] = str(diagnostic_plot_path)
                
                # Create individual XANES plot using plotter
                if "mu_normalized" in processed_data:
                    xanes_plot_path = sample_plot_dir / f"{sample_name}_xanes.png"
                    # Prepare result structure for plotter
                    plot_result = {
                        'sample_name': sample_name,
                        'processed_data': {
                            'energy': processed_data["energy"],
                            'mu_norm': processed_data["mu_normalized"]
                        }
                    }
                    fig = self.plotter.plot_xanes(plot_result, save_path=xanes_plot_path, show_plot=False)
                    plt.close(fig)
                    result["output_files"]["xanes_plot"] = str(xanes_plot_path)

            # Overall success
            result["success"] = True
            result["error"] = None

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            print(f"Error processing {sample_name}: {e}")

        return result

    def process_batch(self,
                     file_paths: List[str | Path],
                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process batch of XAS spectra.

        Parameters
        ----------
        file_paths : list
            List of file paths to process
        progress_callback : callable, optional
            Callback for progress updates

        Returns
        -------
        batch_result : dict
            Batch processing results
        """
        batch_result = {
            "batch_size": len(file_paths),
            "processed_samples": [],
            "timestamp": datetime.now().isoformat()
        }

        # Store normalized spectra for comparison plot
        comparison_data = []

        for i, file_path in enumerate(file_paths):
            sample_name = f"sample_{i+1:03d}"
            result = self.process_single_spectrum(file_path, sample_name)
            batch_result["processed_samples"].append(result)
            
            # Collect data for comparison plot
            if result["success"] and "normalized_csv" in result.get("output_files", {}):
                csv_path = result["output_files"]["normalized_csv"]
                try:
                    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                    comparison_data.append({
                        "sample_name": sample_name,
                        "energy": data[:, 0],
                        "mu_normalized": data[:, 1]
                    })
                except Exception as e:
                    print(f"Warning: Could not load data for comparison plot: {e}")

            if progress_callback:
                progress_callback(i + 1, len(file_paths))

        # Generate comparison plots
        if self.create_diagnostic_plots and len(comparison_data) > 1:
            print("Creating comparison plots...")
            # Create comparison subfolder
            comparison_dir = self.plots_dir / "comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare results dictionary for plotter
            results_dict = {}
            for sample in comparison_data:
                results_dict[sample["sample_name"]] = {
                    'sample_name': sample["sample_name"],
                    'processed_data': {
                        'energy': sample["energy"],
                        'mu_norm': sample["mu_normalized"]
                    }
                }
            
            # XANES comparison plot using plotter
            comparison_plot_path = comparison_dir / "comparison_xanes.png"
            fig = self.plotter.plot_multi_sample_comparison(
                results_dict, 
                plot_type='xanes',
                save_path=comparison_plot_path, 
                show_plot=False
            )
            plt.close(fig)
            batch_result["comparison_plot"] = str(comparison_plot_path)
            print(f"Saved comparison plot: {comparison_plot_path}")

        # Generate quality reports
        print("Generating quality reports...")
        quality_metrics = []
        for result in batch_result["processed_samples"]:
            if result["success"] and "spectrum_quality" in result:
                # Convert dict back to XASSpectrumQualityMetrics object
                try:
                    from .xas_analyzer.spectrum_quality_check import XASSpectrumQualityMetrics
                except ImportError:
                    from xas_analyzer.spectrum_quality_check import XASSpectrumQualityMetrics

                metrics_dict = result["spectrum_quality"]
                # Import the enum
                try:
                    from .xas_analyzer.spectrum_quality_check import XASQualityFlag
                except ImportError:
                    from xas_analyzer.spectrum_quality_check import XASQualityFlag

                quality_flag_str = metrics_dict.get("quality_flag", "acceptable")
                quality_flag = getattr(XASQualityFlag, quality_flag_str.upper(), XASQualityFlag.ACCEPTABLE)

                metrics = XASSpectrumQualityMetrics(
                    sample_id=metrics_dict.get("sample_id", result["sample_name"]),
                    file_path=metrics_dict.get("file_path", ""),
                    classification=metrics_dict.get("classification", "unknown"),
                    confidence=metrics_dict.get("confidence", 0.0),
                    flags=metrics_dict.get("flags", []),
                    data_points=metrics_dict.get("data_points", 0),
                    energy_range=metrics_dict.get("energy_range", 0.0),
                    energy_min=metrics_dict.get("energy_min", 0.0),
                    energy_max=metrics_dict.get("energy_max", 0.0),
                    edge_jump=metrics_dict.get("edge_jump", 0.0),
                    edge_position=metrics_dict.get("edge_position", 0.0),
                    noise_level=metrics_dict.get("noise_level", 0.0),
                    signal_to_noise=metrics_dict.get("signal_to_noise", 0.0),
                    max_intensity=metrics_dict.get("max_intensity", 0.0),
                    saturation_ratio=metrics_dict.get("saturation_ratio", 0.0),
                    deglitching_points_removed=metrics_dict.get("deglitching_points_removed", 0),
                    normalization_quality=metrics_dict.get("normalization_quality", 0.0),
                    pre_edge_slope=metrics_dict.get("pre_edge_slope", 0.0),
                    post_edge_slope=metrics_dict.get("post_edge_slope", 0.0),
                    white_line_intensity=metrics_dict.get("white_line_intensity", 0.0),
                    quality_flag=quality_flag,
                    warnings=metrics_dict.get("warnings", []),
                    suitable_for_analysis=metrics_dict.get("suitable_for_analysis", True)
                )
                quality_metrics.append(metrics)

        if quality_metrics:
            # Generate quality reports
            quality_dir = self.output_dir / "quality_reports"
            quality_dir.mkdir(parents=True, exist_ok=True)

            report_files = generate_xas_quality_report(
                quality_metrics, quality_dir, "xas_batch_quality_report"
            )
            batch_result["quality_reports"] = {k: str(v) for k, v in report_files.items()}

        # ML Analysis (if enabled)
        if self.enable_ml_analysis:
            print("Running ML analysis on batch data...")
            try:
                ml_results = self._run_ml_analysis(batch_result)
                batch_result["ml_analysis"] = ml_results
                if ml_results.get("success"):
                    print(f"ML analysis complete: {ml_results['summary']}")
                else:
                    print(f"ML analysis returned with error: {ml_results.get('error', 'Unknown')}")
            except Exception as e:
                print(f"Warning: ML analysis failed: {e}")
                import traceback
                traceback.print_exc()
                batch_result["ml_analysis"] = {"error": str(e), "success": False}

        # Save batch results (metadata only)
        output_file = self.output_dir / "batch_processing_results.json"
        
        # Custom JSON encoder for numpy arrays (shouldn't be needed anymore, but keep for safety)
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                if isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                return super().default(obj)
        
        with open(output_file, 'w') as f:
            json.dump(batch_result, f, indent=2, cls=NumpyEncoder)

        print(f"Batch processing complete. Results saved to {output_file}")
        return batch_result

    def _run_ml_analysis(self, batch_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run ML analysis on batch preprocessing results.
        
        Parameters
        ----------
        batch_result : dict
            Batch processing results with processed_samples
            
        Returns
        -------
        ml_results : dict
            ML analysis results including PCA, clustering, and trends
        """
        # Step 1: Collect successful samples and extract features
        print("  Step 1/5: Extracting features from normalized spectra...")
        sample_results = []
        
        for result in batch_result["processed_samples"]:
            if not result["success"]:
                continue
                
            # Load normalized data from CSV
            csv_path = result.get("output_files", {}).get("normalized_csv")
            if not csv_path:
                continue
                
            try:
                # Read normalized spectrum
                data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                energy = data[:, 0]
                mu_normalized = data[:, 1]
                
                # Create wrapper object that holds both Pydantic model and numpy arrays
                # (similar to SimplifiedXASSample from test_ml_phase1.py)
                class SampleWrapper:
                    def __init__(self, energy, mu_norm, sample_name, result_dict):
                        self.energy = energy
                        self.normalized_mu = mu_norm
                        self.sample_name = sample_name
                        
                        # Create minimal Pydantic models for feature extractor
                        self.features = None  # Will be populated by extractor
                        self.processing_metadata = ProcessingMetadata(
                            alignment_method=result_dict.get("energy_alignment", {}).get("method", "unknown"),
                            deglitching_points_removed=result_dict.get("deglitching", {}).get("points_removed", 0),
                            quality_score=result_dict.get("spectrum_quality", {}).get("confidence", 0.0)
                        )
                        self.processing_params = XASProcessingParams(
                            e0=result_dict.get("normalization", {}).get("parameters", {}).get("e0", 0.0),
                            edge_step=result_dict.get("normalization", {}).get("parameters", {}).get("edge_step", 0.0)
                        )
                        self.user_metadata = {"file_path": result_dict["file_path"]}
                
                # Create wrapper
                sample_wrapper = SampleWrapper(energy, mu_normalized, result["sample_name"], result)
                
                # Extract features
                features = self.feature_extractor.extract_features(sample_wrapper)
                sample_wrapper.features = features
                
                sample_results.append(sample_wrapper)
                
            except Exception as e:
                print(f"    Warning: Could not extract features from {result['sample_name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(sample_results) < 2:
            return {
                "success": False,
                "error": f"Insufficient samples for ML analysis (need ≥2, got {len(sample_results)})"
            }
        
        print(f"    ✓ Extracted features from {len(sample_results)} samples")
        
        # Step 2: Assemble batch dataset
        print("  Step 2/5: Assembling batch dataset...")
        dataset = self.batch_assembler.assemble_dataset(
            sample_results,
            dataset_id=f"batch_{batch_result['timestamp']}"
        )
        print(f"    ✓ Dataset: {dataset.n_samples} samples × {dataset.n_features} features")
        
        # Step 3: PCA analysis
        print("  Step 3/5: Running PCA analysis...")
        pca_result = self.pca_analyzer.analyze(dataset)
        print(f"    ✓ PCA: {pca_result.n_components} components ({pca_result.variance_captured:.1%} variance)")
        
        # Step 4: Clustering
        print("  Step 4/5: Running clustering analysis...")
        clustering_result = self.clusterer.cluster(dataset, pca_result=pca_result)
        print(f"    ✓ Clustering: {clustering_result.n_clusters} clusters (silhouette={clustering_result.silhouette_score:.3f})")
        
        # Step 5: Trend analysis
        print("  Step 5/5: Running trend analysis...")
        trend_result = self.trend_analyzer.analyze(dataset, clustering_result)
        print(f"    ✓ Trends: {len(trend_result.significant_correlations)} significant correlations")
        
        # Save ML results to separate directory
        ml_output_dir = self.output_dir / "ml_analysis"
        ml_output_dir.mkdir(exist_ok=True)
        
        # Save dataset
        from .xas_ml_modules.xas_batch_assembler import save_dataset_to_json
        dataset_path = ml_output_dir / "dataset.json"
        save_dataset_to_json(dataset, dataset_path)
        
        # Save PCA results
        pca_path = ml_output_dir / "pca_results.json"
        with open(pca_path, 'w') as f:
            f.write(pca_result.model_dump_json(indent=2))
        
        # Save clustering results
        clustering_path = ml_output_dir / "clustering_results.json"
        with open(clustering_path, 'w') as f:
            f.write(clustering_result.model_dump_json(indent=2))
        
        # Save trend results
        trend_path = ml_output_dir / "trend_results.json"
        trend_dict = json.loads(trend_result.model_dump_json())
        with open(trend_path, 'w') as f:
            json.dump(trend_dict, f, indent=2)
        
        print(f"  ✓ ML results saved to {ml_output_dir}")
        
        # Return summary
        return {
            "success": True,
            "output_dir": str(ml_output_dir),
            "files": {
                "dataset": str(dataset_path),
                "pca": str(pca_path),
                "clustering": str(clustering_path),
                "trends": str(trend_path)
            },
            "summary": {
                "n_samples": dataset.n_samples,
                "n_features": dataset.n_features,
                "n_components": pca_result.n_components,
                "variance_captured": f"{pca_result.variance_captured:.1%}",
                "n_clusters": clustering_result.n_clusters,
                "silhouette_score": f"{clustering_result.silhouette_score:.3f}",
                "n_correlations": len(trend_result.significant_correlations),
                "n_outliers": len(trend_result.outlier_indices)
            }
        }


def run_xas_automated_workflow(data_source: str | Path,
                              output_dir: Optional[str] = None,
                              reference_file: Optional[str] = None,
                              create_diagnostic_plots: bool = False,
                              enable_ml_analysis: bool = False) -> Dict[str, Any]:
    """
    Main entry point for automated XAS processing workflow.

    Parameters
    ----------
    data_source : str or Path
        Single file or directory of XAS files
    output_dir : str, optional
        Output directory
    reference_file : str, optional
        Reference spectrum file
    create_diagnostic_plots : bool
        Whether to create diagnostic plots
    enable_ml_analysis : bool
        Whether to run ML analysis on batch data

    Returns
    -------
    result : dict
        Processing results
    """
    processor = XASAutomatedProcessor(output_dir, reference_file, create_diagnostic_plots, enable_ml_analysis)

    data_path = Path(data_source)
    if data_path.is_file():
        # Single file
        result = processor.process_single_spectrum(data_path)
    elif data_path.is_dir():
        # Batch processing
        xas_files = list(data_path.glob("*.dat")) + list(data_path.glob("*.xdi"))
        result = processor.process_batch(xas_files)
    else:
        raise ValueError(f"Invalid data source: {data_source}")

    return result


# Convenience functions for agent framework integration

def run_xas_workflow(data_dir: str | Path,
                    output_dir: Optional[str | Path] = None,
                    pattern: str = "*",
                    **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run complete XAS workflow.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing XAS data files
    output_dir : str or Path, optional
        Output directory for results
    pattern : str
        File pattern for batch processing
    **kwargs
        Additional arguments passed to XASAutomatedProcessor

    Returns
    -------
    results : dict
        Workflow results
    """
    workflow = XASAutomatedProcessor(output_dir=output_dir, **kwargs)
    data_path = Path(data_dir)
    
    if data_path.is_file():
        return workflow.process_single_spectrum(data_path)
    elif data_path.is_dir():
        xas_files = list(data_path.glob(f"{pattern}.dat")) + list(data_path.glob(f"{pattern}.xdi"))
        return workflow.process_batch(xas_files)
    else:
        raise ValueError(f"Invalid data source: {data_dir}")


def analyze_single_xas_file(file_path: str | Path,
                          output_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Analyze a single XAS file.

    Parameters
    ----------
    file_path : str or Path
        Path to XAS file
    output_dir : str or Path, optional
        Output directory

    Returns
    -------
    results : dict
        Analysis results
    """
    processor = XASAutomatedProcessor(output_dir=output_dir)
    return processor.process_single_spectrum(file_path)


# Command-line interface for testing
if __name__ == "__main__":
    import argparse

    # Default data directory for testing
    default_data_dir = Path(__file__).resolve().parents[2] / "project_root" / "xas_raw_data"

    parser = argparse.ArgumentParser(description="XAS Analysis Workflow")
    parser.add_argument("data_dir", nargs='?', default=str(default_data_dir),
                       help="Directory containing XAS files (default: project_root/xas_raw_data)")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--pattern", "-p", default="*", help="File pattern")

    args = parser.parse_args()

    print(f"Running XAS analysis on: {args.data_dir}")
    results = run_xas_workflow(args.data_dir, args.output, pattern=args.pattern, create_diagnostic_plots=True)
    print(f"Analysis complete! Results saved to: {results['output_dir'] if 'output_dir' in results else 'output directory'}")
    print(f"Processed {len(results['processed_samples']) if 'processed_samples' in results else 'samples'} samples")