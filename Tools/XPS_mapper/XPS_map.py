"""
# xps_map_processor.py
# Standalone processor for two cases map files:
# after xps_reader routing:
#  Case 1: single-energy maps - a 2D array map of intensities with no energy dimension
#  Case 2: hyperspectral maps - a 3D data cube with energy axis (a spectrum per pixel)

Features:
- Multi-format support (PHI MultiPak, ASCII)
- Automatic format detection
- Robust error handling
- Standardized data structures
- Comprehensive validation
"""
from __future__ import annotations

import os
import re
import time
import yaml
import traceback
import numpy as np  
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Union
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import sys
from pathlib import Path

# Optional: pymcr (handled in chemometrics_utils)
try:
    from pymcr import MCR
except ImportError:
    MCR = None

# Add the project root directory to Python path
project_root = Path(__file__).parents[2] / "project_root"  # Go up to zzy_llm/project_root
tools_dir = Path(__file__).parent.parent  # Go up to Tools directory

# Add paths for imports
if str(project_root.parent) not in sys.path:
    sys.path.append(str(project_root.parent))
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

# Add XPS_Fitter to path for XPS_peakfitting_V2
xps_fitter_path = tools_dir / "XPS_Fitter"
if str(xps_fitter_path) not in sys.path:
    sys.path.insert(0, str(xps_fitter_path))

from tool_utils import (
    format_time,
    load_yaml_settings
)

# Import XPS fitting functions
try:
    from XPS_peakfitting_V2 import (
        load_yaml_template,
        fit_region_with_template,
        parse_template_to_regions
    )
except (ImportError, UnicodeEncodeError) as e:
    print(f"Warning: Could not import XPS_peakfitting_V2: {e}")
    load_yaml_template = None
    fit_region_with_template = None
    parse_template_to_regions = None

# Import modular processing functions
from case1_2d_processing import (
    compute_net_and_ratio,
    denoise_map,
    threshold_segment,
    morph_cleanup,
    roi_stats,
    pearson_correlation,
    process_2d_map
)

from case2_hyperspectral_processing import (
    baseline_als,
    estimate_energy_shift,
    gaussian_model,
    fit_average_spectrum,
    fit_pixel_spectrum,
    process_hyperspectral,
    nnls_project_pixel,
    pca_cluster_analysis
)

# Import visualization functions
from map_plots_basic import (
    ensure_output_dir,
    save_fig,
    plot_2d_overview,
    plot_area_maps,
    plot_shift_mse_maps,
    plot_average_spectrum
)

from component_plots import (
    plot_pca_components,
    plot_pca_scores,
    plot_pca_loadings,
    plot_nmf_components,
    plot_scree
)

from cluster_plots import (
    plot_cluster_analysis,
    plot_cluster_map,
    plot_cluster_spectra,
    plot_cluster_scatter,
    plot_dendrogram,
    plot_cluster_size_distribution
)

from chemometrics_plots import (
    plot_spectra_waterfall,
    plot_pre_image,
    plot_mcr_components
)

# Import report generator
from report_generator import (
    AnalysisReport,
    generate_comprehensive_report,
    create_pca_report_data,
    create_mcr_report_data,
    create_cluster_report_data
)

# ========== CONFIGURATION ==========
# Use project_root structure for paths
PROJECT_ROOT = Path(__file__).parents[2] / "project_root"
RAW_DATA_DIR = PROJECT_ROOT / "00_raw_data"
OUTPUT_DIR = PROJECT_ROOT / "05_map_data"  #  map output directory
CONFIG_FILE = PROJECT_ROOT / "xps_config" / "region_definitions.yaml"
TEMPLATE_DIR = PROJECT_ROOT / "xps_config" / "LIB_fit_template"
RAW_FILE_PATTERNS = ["*.txt", "*.asc", "*.dat", "*.csv"]

# Core dependencies for PCA/clustering
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.optimize import nnls

# Import chemometrics utilities from separate module
from chemometrics_utils import (
    compute_pre_image,
    normalize_l1,
    mask_low_counts,
    charge_align_cube,
    run_mcr_on_cube,
    run_mcr_with_pca_init,
    compute_spectral_variability,
    MCR_AVAILABLE
)

# Import spatial masking utilities
from apply_mask import (
    apply_intensity_mask,
    apply_pca_score_mask,
    apply_cluster_mask,
    validate_and_mask_clusters,
    save_mask_visualization,
    combine_masks,
    get_mask_from_config,
    MaskingResults
)

# Import MCR fitting and quantification functions
from mcr_fitting import (
    fit_mcr_components,
    save_mcr_fitting_results,
    save_quantitative_concentration_maps,
    plot_mcr_fitting_results,
    plot_atomic_percentages,
    plot_quantitative_concentration_maps,
    plot_peak_parameter_summary,
    plot_combined_concentration_maps
)

# Import map parser module
from map_parser import (
    detect_and_parse,
    count_hyperspec_rows_in_file,
    Map2D,
    HyperspectralMap,
    MapMetadata
)

# Matplotlib (headless mode for server environments)
import matplotlib
matplotlib.use("Agg")

# Logging (minimal INFO by default)
logger = logging.getLogger("xps_map")
def setup_logger(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(message)s", level=level)

# ========== CONFIGURATION LOADER ==========
def resolve_region_definitions(config: Dict) -> Dict[str, Dict]:
    return config.get('regions', {})

def load_xps_config(config_path: str = CONFIG_FILE) -> Tuple[Dict, Dict[str, Dict]]:
    """Load YAML configuration and region definitions."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. Please ensure the YAML config exists."
        )
    try:
        config = load_yaml_settings(config_path)
        region_defs = resolve_region_definitions(config)
        logger.info(f"Loaded config: {config_path.name}")
        if region_defs:
            logger.info(f"Regions defined: {', '.join(region_defs.keys())}")
        cal = config.get('energy_calibration', {})
        if cal.get('enable', True):
            ref_region = cal.get('reference_region', 'C1s')
            target_be = cal.get('target_binding_energy_ev', 284.8)
            logger.info(f"Energy calibration: {ref_region} → {target_be} eV")
        return config, region_defs
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in config file: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load config: {e}")

def ensure_output_directories():
    """Ensure all output directories exist."""
    directories = [
        OUTPUT_DIR,
        OUTPUT_DIR / "analysis_results" / "2d_maps",
        OUTPUT_DIR / "analysis_results" / "hyperspectral_maps",
        OUTPUT_DIR / "plots" / "2d_maps",
        OUTPUT_DIR / "plots" / "hyperspectral_maps"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def load_region_definitions(config_path: str = CONFIG_FILE) -> Dict:
    _, region_defs = load_xps_config(config_path)
    return region_defs

# Data structures now imported from map_parser module

# ============================ Utility functions ============================
def safe_tag(s: Optional[str]) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s or "unknown")

# Helper: safe dict getter to avoid attribute errors on numpy arrays
def _safe_get(obj, key, default=None):
    """Safely get a key from a dict-like object. Returns default if obj isn't a dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

# Parsing functions now imported from map_parser module

# ============================ Map processing wrappers ============================

def parse_map_with_config(file_path: str,
                          override_nx: Optional[int] = None,
                          override_ny: Optional[int] = None) -> Union[Map2D, HyperspectralMap]:
    """
    Wrapper around map_parser.detect_and_parse that provides region definitions from config.
    
    Args:
        file_path: Path to the data file
        override_nx: Optional override for number of x points
        override_ny: Optional override for number of y points
        
    Returns:
        Parsed Map2D or HyperspectralMap object
    """
    # Load region definitions from config
    region_defs = load_region_definitions(CONFIG_FILE)
    
    # Call parser with region definitions for energy axis generation
    return detect_and_parse(
        file_path=file_path,
        override_nx=override_nx,
        override_ny=override_ny,
        region_definitions=region_defs
    )


from typing import Any


def summarize_map_spectra(file_path: str,
                          override_nx: Optional[int] = None,
                          override_ny: Optional[int] = None) -> Dict[str, Any]:
    """
    Summarize how many spectra exist in the map:
    - Hyperspectral: spectra_count = nx * ny, report nE, and estimate actual row count in file.
    - Single-energy map: spectra_count = 0 (no per-pixel spectra), report pixel count.
    Returns a dict with fields:
      type: 'hyperspectral' or '2d_single_energy'
      spectra_count: int
      nx, ny: ints
      nE: int or None
      pixel_count: int
      region: str
      energy_min, energy_max: float or None
      actual_spectra_rows_in_file: int or None (for hyperspectral)
    """
    parsed = parse_map_with_config(file_path, override_nx=override_nx, override_ny=override_ny)

    if isinstance(parsed, HyperspectralMap):
        ny, nx, nE = parsed.shape
        spectra_rows = count_hyperspec_rows_in_file(file_path)
        return {
            "type": "hyperspectral",
            "region": parsed.metadata.region,
            "nx": nx,
            "ny": ny,
            "pixel_count": nx * ny,
            "spectra_count": nx * ny,
            "nE": int(nE),
            "energy_min": float(parsed.energy.min()) if parsed.energy.size else None,
            "energy_max": float(parsed.energy.max()) if parsed.energy.size else None,
            "actual_spectra_rows_in_file": spectra_rows
        }
    elif isinstance(parsed, Map2D):
        ny, nx = parsed.shape
        return {
            "type": "2d_single_energy",
            "region": parsed.metadata.region,
            "nx": nx,
            "ny": ny,
            "pixel_count": nx * ny,
            "spectra_count": 0,
            "nE": None,
            "energy_min": None,
            "energy_max": None,
            "actual_spectra_rows_in_file": None
        }
    else:
        return {
            "type": "unknown",
            "region": None,
            "nx": None,
            "ny": None,
            "pixel_count": None,
            "spectra_count": None,
            "nE": None,
            "energy_min": None,
            "energy_max": None,
            "actual_spectra_rows_in_file": None
        }


def log_map_spectra_summary(file_path: str,
                            override_nx: Optional[int] = None,
                            override_ny: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience: summarize and log one-line message.
    Returns the summary dict.
    """
    info = summarize_map_spectra(file_path, override_nx=override_nx, override_ny=override_ny)
    if info["type"] == "hyperspectral":
        msg = (f"Spectra: {info['spectra_count']} ({info['ny']}×{info['nx']} pixels), "
               f"nE={info['nE']}, E=[{info['energy_min']:.2f}, {info['energy_max']:.2f}] eV")
        if info.get("actual_spectra_rows_in_file") is not None:
            msg += f", rows_in_file≈{info['actual_spectra_rows_in_file']}"
        logger.info(msg)
    elif info["type"] == "2d_single_energy":
        logger.info(f"Single-energy map (no per-pixel spectra). Pixels: {info['pixel_count']} "
                    f"({info['ny']}×{info['nx']})")
    else:
        logger.info("Unrecognized map type.")
    return info

# ============================ Case 1: 2D map analysis ============================
# Core Case 1 functions are in case1_2d_processing.py and imported 
# Functions include: compute_net_and_ratio, denoise_map, threshold_segment, 
#                   morph_cleanup, roi_stats, pearson_correlation, process_2d_map

# ============================ Case 2: Hyperspectral analysis ============================
# Core Case 2 functions are in case2_hyperspectral_processing.py and imported 
# Functions include: baseline_als, estimate_energy_shift, gaussian_model,
#                   fit_average_spectrum, fit_pixel_spectrum, process_hyperspectral,
#                   nnls_project_pixel, pca_cluster_analysis

# fit_mcr_components is now imported from mcr_fitting module and should not be redefined here


def pca_cluster_preselect(hmap: HyperspectralMap,
                          n_pca: int = 3,
                          n_clusters: int = 4,
                          use_minibatch: bool = True,
                          normalize: str = "l2",
                          output_dir: Optional[Path] = None) -> Dict:
    """
    Perform PCA on pixel spectra and cluster scores to find spectral phases.
    
    Args:
        hmap: HyperspectralMap object containing the data
        n_pca: Number of PCA components to extract
        n_clusters: Number of clusters for KMeans
        use_minibatch: Whether to use MiniBatchKMeans
        normalize: Normalization method ('l2', 'mean', or None)
        output_dir: Directory to save PCA visualization plots
        
    Returns:
        dict: { 'pca': {...}, 'labels': (ny,nx), 'cluster_info': list[dict] }
    """
    ny, nx, nE = hmap.shape
    X = hmap.cube.reshape(ny*nx, nE).astype(float)
    X_original = X.copy()  # Keep original unnormalized data for cluster mean spectra
    energy = hmap.energy  # Get energy array for plotting

    # Normalize spectra for PCA stability
    if normalize == "l2":
        norms = np.linalg.norm(X, axis=1) + 1e-12
        X = X / norms[:, None]
    elif normalize == "mean":
        means = np.mean(X, axis=1) + 1e-12
        X = X / means[:, None]

    # PCA
    pca = PCA(n_components=n_pca, whiten=False, random_state=0)
    scores = pca.fit_transform(X)  # shape (N, n_pca)
    explained = pca.explained_variance_ratio_
    score_maps = scores.reshape(ny, nx, n_pca)
    
    # Print PCA results
    print("\nPCA Analysis Results:")
    print(f"Number of components: {n_pca}")
    print("Explained variance ratios:")
    for i, var in enumerate(explained):
        print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")
        
    # Visualization using plotting functions
    if output_dir is not None:
        base_name = hmap.name if hasattr(hmap, 'name') else 'map'
        pca_info = {
            'components': pca.components_,
            'score_maps': score_maps,
            'explained_variance': explained
        }
        plot_pca_components(energy, pca_info, region="", base_name=base_name, 
                           output_dir=output_dir, show=False)

    # Clustering
    if use_minibatch:
        # Use larger batch_size to prevent memory leak on Windows with MKL
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=2048)
    else:
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    labels_1d = km.fit_predict(scores)
    labels = labels_1d.reshape(ny, nx)
    
    # Print clustering results
    print("\nClustering Results:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} pixels ({count/(ny*nx)*100:.1f}% of data)")

    # Representative spectra (mean and medoid index in PCA space)
    cluster_info = []
    for k in range(n_clusters):
        idx = np.where(labels_1d == k)[0]
        if idx.size == 0:
            cluster_info.append({
                "cluster": k, "size": 0, "mean_spec": None, "medoid_index": None,
                "centroid_scores": None
            })
            continue
        # mean spectrum in energy domain (use original unnormalized data for proper intensity scale)
        mean_spec = np.mean(X_original[idx, :], axis=0)
        # medoid: closest to centroid in PCA score space
        centroid = np.mean(scores[idx, :], axis=0)
        dists = np.linalg.norm(scores[idx, :] - centroid[None, :], axis=1)
        medoid_local = np.argmin(dists)
        medoid_index = int(idx[medoid_local])
        cluster_info.append({
            "cluster": k,
            "size": int(idx.size),
            "mean_spec": mean_spec,
            "medoid_index": medoid_index,
            "centroid_scores": centroid
        })

    return {
        "pca": {
            "components": pca.components_,
            "explained_variance": explained,
            "score_maps": score_maps
        },
        "labels": labels,
        "cluster_info": cluster_info
    }



# ============================ Visualization utilities ============================
# ensure_output_dir and save_fig now imported from map_plots_basic

# Visualization functions now imported from modular files:
# - plot_2d_overview from map_plots_basic
# - plot_average_spectrum, plot_area_maps, plot_shift_mse_maps from map_plots_basic
# - plot_pca_components, plot_nmf_components from component_plots
# - plot_cluster_analysis from cluster_plots

# ============================ High-level processing entry ============================
def process_xps_map(
    file_path: str,
    off_peak_path: Optional[str] = None,
    case1_sigma: float = 1.0,
    case1_threshold_method: str = "otsu",
    case1_percentile: float = 95.0,
    case1_morph_op: str = "open",
    case1_morph_size: int = 2,
    init_peaks: Optional[List[Tuple[float, float]]] = None,
    bg_lam: float = 1e5,
    bg_p: float = 0.01,
    bg_niter: int = 10,
    max_shift_bins: int = 10,
    do_pca: bool = True,
    n_pca: int = 3,
    do_nmf: bool = True,
    n_nmf: int = 3,
    make_plots: bool = True,
    show_plots: bool = False,
    override_nx: Optional[int] = None, 
    override_ny: Optional[int] = None,
    output_dir: Optional[Path] = None,  # New parameter for output directory
) -> Dict[str, Union[Map2D, HyperspectralMap, Dict]]:
    parsed = parse_map_with_config(file_path, override_nx=override_nx, override_ny=override_ny)
    
    # Use provided output_dir or default to OUTPUT_DIR
    if output_dir is None:
        output_path = Path(OUTPUT_DIR)
    else:
        output_path = Path(output_dir)
    
    # Ensure output directories exist
    ensure_output_directories()
    
    # Create subdirectories based on data type
    if isinstance(parsed, Map2D):
        specific_output = output_path / "analysis_results" / "2d_maps"
        plots_output = output_path / "plots" / "2d_maps"
    else:
        specific_output = output_path / "analysis_results" / "hyperspectral_maps" 
        plots_output = output_path / "plots" / "hyperspectral_maps"
        
    specific_output.mkdir(parents=True, exist_ok=True)
    plots_output.mkdir(parents=True, exist_ok=True)

    if isinstance(parsed, Map2D):
        off_map_parsed = parse_map_with_config(off_peak_path) if off_peak_path else None
        norm = compute_net_and_ratio(parsed, off_map_parsed)
        base_img = norm["net"] if norm["ratio"] is not None else parsed.data
        denoised = denoise_map(base_img, sigma=case1_sigma)
        mask, thr = threshold_segment(denoised, method=case1_threshold_method, percentile=case1_percentile)
        mask_clean = morph_cleanup(mask, op=case1_morph_op, size=case1_morph_size)
        stats = roi_stats(denoised, mask_clean)

        # Save data products to 2D maps directory
        save_2d_results(parsed, denoised, mask_clean, stats, Path(file_path), specific_output)
        # Plots to plots/2d_maps directory
        if make_plots:
            plot_2d_overview(parsed, denoised, mask_clean, norm["net"], norm["ratio"], thr, plots_output, show_plots)

        outputs = {
            "parsed": parsed,
            "net_map": norm["net"],
            "ratio_map": norm["ratio"],
            "denoised": denoised,
            "threshold": thr,
            "mask": mask_clean,
            "roi_stats": stats,
            "output_dir": specific_output
        }
        return outputs

    elif isinstance(parsed, HyperspectralMap):
        if not init_peaks:
            raise ValueError("init_peaks must be provided for hyperspectral fitting (e.g., [(284.8, 0.3), (285.3, 0.3)])")
        results = process_hyperspectral(
            parsed, init_peaks=init_peaks,
            background_lam=bg_lam, background_p=bg_p, niter_bg=bg_niter,
            max_shift_bins=max_shift_bins, do_pca=do_pca, n_pca=n_pca,
            do_nmf=do_nmf, n_nmf=n_nmf
        )
        # Save products to hyperspectral maps directory
        save_hyperspectral_results(parsed, results, Path(file_path), specific_output)
        # Plots to plots/hyperspectral_maps directory
        if make_plots:
            # Use modular plotting functions
            base_name = Path(parsed.metadata.source_file).stem if parsed.metadata.source_file else "map"
            region = parsed.metadata.region or "unknown"
            
            # Plot average spectrum
            plot_average_spectrum(
                parsed.energy, 
                results["avg_spectrum"],
                results.get("avg_spectrum_baseline", np.zeros_like(parsed.energy)),
                results.get("avg_spectrum_corrected", results["avg_spectrum"]),
                region, plots_output, base_name, show_plots
            )
            
            # Plot area maps
            if "area_maps" in results:
                plot_area_maps(results["area_maps"], plots_output, base_name, region, show_plots)
            
            # Plot shift and MSE maps
            if "shift_map" in results and "mse_map" in results:
                plot_shift_mse_maps(
                    results["shift_map"], results["mse_map"],
                    plots_output, base_name, region, show_plots
                )
            
            # Plot PCA components
            if "pca" in results:
                pca_info = results["pca"]
                plot_pca_components(
                    parsed.energy, pca_info["components"],
                    pca_info["score_maps"], pca_info["explained_variance"],
                    plots_output, base_name, region, show_plots
                )
            
            # Plot NMF components
            if "nmf" in results:
                nmf_info = results["nmf"]
                plot_nmf_components(
                    parsed.energy, nmf_info["components"],
                    nmf_info["abundance_maps"],
                    plots_output, base_name, region, show_plots
                )

            # Plot MCR components if present in results
            if "mcr_results" in results and results.get("mcr_results"):
                mcr = results["mcr_results"]
                comp_spec = mcr.get("component_spectra")
                conc_maps = mcr.get("conc_maps")
                if comp_spec is not None and conc_maps is not None:
                    try:
                        plot_mcr_components(
                            parsed.energy,
                            comp_spec,
                            conc_maps,
                            mcr.get("method", "MCR"),
                            plots_output,
                            base_name=base_name,
                            region=region,
                            x_axis=parsed.x_axis,
                            y_axis=parsed.y_axis,
                            show=show_plots
                        )
                    except Exception as e:
                        logger.warning(f"Failed to plot MCR components: {e}")
                    
                    # Plot MCR fitting results if available
                    fitted_components = mcr.get('fitted_components')
                    logger.info(f"DEBUG: fitted_components present: {fitted_components is not None}")
                    logger.info(f"DEBUG: plots_output directory: {plots_output}")
                    if fitted_components:
                        try:
                            logger.info("DEBUG: Starting MCR fitting result plots...")
                            # 1. Fitted component spectra with peak deconvolution
                            logger.info("DEBUG: Plotting fitted spectra...")
                            plot_mcr_fitting_results(
                                mcr, parsed.energy, plots_output,
                                base_name, region, show=show_plots
                            )
                            
                            # 2. Atomic percentage bar chart and pie chart
                            logger.info("DEBUG: Plotting atomic percentages...")
                            plot_atomic_percentages(
                                fitted_components, plots_output,
                                base_name, region, 
                                method=mcr.get("method", "MCR"),
                                show=show_plots
                            )
                            
                            # 3. Quantitative concentration maps
                            logger.info(f"DEBUG: Calling plot_quantitative_concentration_maps with plots_output={plots_output}")
                            plot_quantitative_concentration_maps(
                                mcr, plots_output, base_name, region,
                                x_axis=parsed.x_axis,
                                y_axis=parsed.y_axis,
                                show=show_plots
                            )
                            logger.info("DEBUG: plot_quantitative_concentration_maps completed")
                            
                            # 4. Peak parameter summary table
                            logger.info("DEBUG: Plotting peak parameter summary...")
                            plot_peak_parameter_summary(
                                fitted_components, plots_output,
                                base_name, region, show=show_plots
                            )
                            
                            # 5. Combined concentration map (all species side-by-side)
                            logger.info("DEBUG: Plotting combined concentration map...")
                            plot_combined_concentration_maps(
                                mcr, plots_output, base_name, region,
                                x_axis=parsed.x_axis,
                                y_axis=parsed.y_axis,
                                show=show_plots
                            )
                            
                            logger.info(f"Generated MCR fitting visualization plots for {region}")
                        except Exception as e:
                            logger.error(f"EXCEPTION in MCR fitting plots: {e}")
                            import traceback
                            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        outputs = {
            "parsed": parsed,
            "hyperspectral_results": results,
            "output_dir": specific_output
        }
        return outputs

    else:
        raise RuntimeError("Unknown parsed type")
# Process files in the main loop above

# ============================ Batch processing utilities ============================
def find_map_files(data_dir: str = RAW_DATA_DIR, patterns: List[str] = RAW_FILE_PATTERNS) -> List[Path]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = []
    for pattern in patterns:
        files.extend(data_path.glob(pattern))
    return sorted(files)

# ============================ Simple helpers ============================
def get_initial_peaks(region: str, region_defs: Dict) -> Optional[List[Tuple[float, float]]]:
    if not region or not region_defs:
        return None
    if region in region_defs:
        region_info = region_defs[region]
        components = region_info.get('components', [])
        if components:
            peaks = []
            for comp in components:
                center = comp.get('binding_energy_ev')
                sigma = comp.get('fwhm_ev', 0.6) / 2.355
                if center is not None:
                    peaks.append((float(center), float(sigma)))
            if peaks:
                return peaks
    region_lower = region.lower()
    for key, value in region_defs.items():
        if key.lower() == region_lower:
            components = value.get('components', [])
            if components:
                peaks = []
                for comp in components:
                    center = comp.get('binding_energy_ev')
                    sigma = comp.get('fwhm_ev', 0.6) / 2.355
                    if center is not None:
                        peaks.append((float(center), float(sigma)))
                if peaks:
                    return peaks
    return None

# ============================ Processing functions (minimal prints) ============================


def process_2d_map(parsed: Map2D, file_path: Path, analysis_output_dir: Path, plots_output_dir: Path, make_plots: bool = True, show_plots: bool = False) -> Dict:
    denoised = denoise_map(parsed.data, sigma=1.0)
    mask, threshold = threshold_segment(denoised, method="otsu")
    mask_clean = morph_cleanup(mask, op="open", size=2)
    stats = roi_stats(denoised, mask_clean)
    ensure_output_dir(analysis_output_dir)
    ensure_output_dir(plots_output_dir)
    save_2d_results(parsed, denoised, mask_clean, stats, file_path, analysis_output_dir)
    if make_plots:
        plot_2d_overview(parsed, denoised, mask_clean, None, None, threshold, plots_output_dir, show_plots)
    return {'parsed': parsed, 'denoised': denoised, 'mask': mask_clean, 'threshold': threshold, 'roi_stats': stats}

# Fitting functions removed - low S/N ratio makes fitting unreliable on map data
# Use PCA/NMF/clustering analysis instead via process_hyperspectral

def process_hyperspectral_map_simple(
    map_data: HyperspectralMap,
    output_dir: Path,
    config: Dict = None,
    make_plots: bool = True,
    show_plots: bool = False,
    n_pca: int = 3,
    n_clusters: int = 4,
    do_nmf: bool = True,
    n_nmf: int = 3,
    compute_pre: bool = True,
    charge_ref_energy: Optional[float] = None,
    charge_window: float = 1.0,
    mask_threshold: Optional[float] = None,
    normalize_spectra: bool = False,
    do_mcr: bool = False,
    n_mcr: int = 3,
    validate_clusters: bool = True,
    min_fwhm: float = 0.5,
    cluster_mask: Optional[list] = None
) -> Dict:
    """
    Process hyperspectral map data using chemometrics workflow.
    Includes: PRE, charge correction, masking, cluster validation, PCA, MCR-ALS, clustering.
    
    Args:
        map_data: HyperspectralMap object
        output_dir: Directory for plots
        config: Configuration dictionary with region definitions (for element-specific ranges)
        compute_pre: Calculate pattern recognition entropy
        charge_ref_energy: Reference BE for alignment (e.g., 284.8 for C1s)
        charge_window: Search window around ref energy (eV)
        mask_threshold: Minimum total counts per pixel (None = no masking)
        normalize_spectra: Apply L1 normalization
        do_mcr: Run MCR-ALS analysis
        n_mcr: Number of MCR components
        validate_clusters: Run cluster validation to reject outliers/artifacts
        cluster_mask: List of cluster IDs to keep (e.g., [2] to keep only cluster 2)
                      If provided, performs initial clustering, then masks out unwanted clusters
        min_fwhm: Minimum FWHM (eV) for cluster validation
    """
    ny, nx = map_data.shape[:2]
    energy = map_data.energy
    cube = map_data.cube.copy()  # Work with copy
    
    logger.info(f"Processing hyperspectral map: {ny}x{nx} pixels, {len(energy)} energy points")
    logger.info(f"Energy range: {energy.min():.1f}-{energy.max():.1f} eV")
    logger.info(f"Intensity range: {cube.min():.1f}-{cube.max():.1f}")
    
    # Step 0: Initial data inspection - waterfall/overlay plot
    variability_metrics = None
    if make_plots:
        logger.info("Generating initial spectra inspection plot...")
        variability = plot_spectra_waterfall(
            cube, energy, output_dir,
            base_name=map_data.name,
            region=map_data.metadata.region or "",
            n_spectra=min(20, ny*nx),
            plot_mode="overlay",
            show=show_plots
        )
        variability_metrics = variability
        logger.info(f"Spectral variability: mean σ={variability['mean_std']:.1f}, "
                   f"max σ={variability['max_std']:.1f}")
    else:
        # Compute variability even if not plotting
        variability_metrics = compute_spectral_variability(cube)
        logger.info(f"Spectral variability: mean σ={variability_metrics['mean_std']:.1f}, "
                   f"max σ={variability_metrics['max_std']:.1f}")
    
    # Chemometrics preprocessing
    mask = None
    pre_image = None
    mcr_results = None
    
    # 1. PRE (Pattern Recognition Entropy) analysis
    if compute_pre:
        pre_image = compute_pre_image(cube)
        logger.info(f"PRE computed: range [{pre_image.min():.2f}, {pre_image.max():.2f}]")
        
        # Visualize PRE image
        if make_plots:
            plot_pre_image(
                pre_image, output_dir,
                base_name=map_data.name,
                region=map_data.metadata.region or "",
                x_axis=map_data.x_axis,
                y_axis=map_data.y_axis,
                show=show_plots
            )
    
    # 2. Mask low-count pixels
    if mask_threshold is not None:
        mask = mask_low_counts(cube, mask_threshold)
        valid_pixels = np.sum(mask)
        logger.info(f"Masking applied: {valid_pixels}/{ny*nx} pixels above threshold {mask_threshold}")
        cube[~mask] = 0.0
    
    # 3. Charge correction (align spectra)
    if charge_ref_energy is not None:
        logger.info(f"Charge correcting to {charge_ref_energy} eV (±{charge_window} eV window)")
        cube = charge_align_cube(cube, energy, charge_ref_energy, charge_window)
    
    # 3.5. Energy range selection and intensity masking based on config
    # This step allows focusing analysis on specific energy regions and masking artifacts
    # First try to read element-specific range from region definition, fall back to global config
    region_name = map_data.metadata.region
    region_def = None
    if config and region_name and 'regions' in config:
        region_def = config['regions'].get(region_name, None)
    
    # Get analysis range (element-specific preferred, global fallback)
    energy_range = None
    if region_def and 'analysis_range' in region_def:
        energy_range = region_def['analysis_range']
        logger.info(f"Using element-specific analysis_range for {region_name}: {energy_range}")
    elif config:
        energy_range = config.get('analysis_energy_range', None)
        if energy_range:
            logger.info(f"Using global analysis_energy_range: {energy_range}")
    
    # Get intensity mask threshold (element-specific preferred, global fallback)
    intensity_mask_threshold = None
    if region_def and 'intensity_mask_threshold' in region_def:
        intensity_mask_threshold = region_def['intensity_mask_threshold']
        if intensity_mask_threshold is not None:
            logger.info(f"Using element-specific intensity_mask_threshold for {region_name}: {intensity_mask_threshold}")
        else:
            logger.info(f"Element-specific intensity_mask_threshold is null for {region_name} - no masking")
    elif config:
        intensity_mask_threshold = config.get('intensity_mask_threshold', None)
        if intensity_mask_threshold is not None:
            logger.info(f"Using global intensity_mask_threshold: {intensity_mask_threshold}")
    
    # Get cluster mask (element-specific preferred, global fallback)
    # This allows focusing analysis only on specific clusters (e.g., [2] to keep cluster 2)
    logger.info(f"DEBUG: cluster_mask parameter = {cluster_mask}")
    logger.info(f"DEBUG: region_name = {region_name}")
    logger.info(f"DEBUG: region_def = {region_def is not None}")
    if region_def:
        logger.info(f"DEBUG: region_def keys = {list(region_def.keys())}")
        logger.info(f"DEBUG: 'focus_clusters' in region_def = {'focus_clusters' in region_def}")
    
    if cluster_mask is None:  # Only read from config if not explicitly provided
        if region_def and 'focus_clusters' in region_def:
            cluster_mask = region_def['focus_clusters']
            logger.info(f"Using element-specific focus_clusters for {region_name}: {cluster_mask}")
        elif config and 'focus_clusters' in config:
            cluster_mask = config['focus_clusters']
            logger.info(f"Using global focus_clusters: {cluster_mask}")
    
    if energy_range is not None:
        e_min, e_max = energy_range
        logger.info(f"Restricting analysis to energy range: {e_min}-{e_max} eV")
        
        # Find indices for energy range
        energy_mask = (energy >= e_min) & (energy <= e_max)
        energy_indices = np.where(energy_mask)[0]
        
        if len(energy_indices) == 0:
            logger.warning(f"No energy points found in range {e_min}-{e_max} eV")
        else:
            # Crop cube and energy to selected range
            cube = cube[:, :, energy_indices]
            energy = energy[energy_indices]
            logger.info(f"Energy range cropped: {len(energy)} points from {energy.min():.1f} to {energy.max():.1f} eV")
    
    if intensity_mask_threshold is not None:
        logger.info(f"✓ Applying intensity masking: threshold = {intensity_mask_threshold}")
        
        # Calculate mean spectrum across all pixels
        mean_spectrum = np.mean(cube, axis=(0, 1))
        
        # Create mask for energy points with sufficient intensity
        intensity_mask_energy = mean_spectrum > intensity_mask_threshold
        
        if np.sum(intensity_mask_energy) == 0:
            logger.warning(f"No energy points above intensity threshold {intensity_mask_threshold}")
        else:
            # Apply mask to cube - zero out low-intensity energy channels
            cube[:, :, ~intensity_mask_energy] = 0.0
            masked_points = np.sum(~intensity_mask_energy)
            logger.info(f"Masked {masked_points}/{len(energy)} energy points with low intensity")
    
    # 4. Cluster validation (optional outlier detection before MCR)
    cluster_validation_results = None
    if validate_clusters:
        logger.info("Running cluster validation...")
        from chemometrics_plots import plot_cluster_validation
        
        # Create temporary HyperspectralMap for clustering
        temp_map = HyperspectralMap(
            cube=cube.copy(),
            energy=energy,
            metadata=map_data.metadata,
            name=map_data.name
        )
        
        # Quick clustering with fewer components
        quick_cluster = pca_cluster_analysis(
            temp_map,
            n_pca=min(3, n_pca),
            n_clusters=n_clusters,
            normalize='l1' if normalize_spectra else 'l2'
        )
        
        # Validate clusters using apply_mask module
        valid_mask, validation_info = validate_and_mask_clusters(
            cube=cube,
            energy=energy,
            cluster_results=quick_cluster,
            region=map_data.metadata.region or "Unknown",
            min_fwhm=min_fwhm,
            silhouette_threshold=0.2
        )
        
        cluster_validation_results = {
            'cluster_results': quick_cluster,
            'valid_mask': valid_mask,
            'validation_info': validation_info
        }
        
        # Visualize validation
        if make_plots:
            plot_cluster_validation(
                quick_cluster,
                energy,
                valid_mask,
                output_dir,
                base_name=map_data.name,
                region=map_data.metadata.region or "",
                show=show_plots
            )
            
            # Save validation mask visualization
            save_mask_visualization(
                valid_mask,
                (ny, nx),
                output_dir,
                map_data.name,
                'validation',
                show=show_plots
            )
        
        # Apply mask to cube (zero out invalid pixels)
        n_invalid = np.sum(~valid_mask)
        if n_invalid > 0:
            logger.info(f"Masking {n_invalid} invalid pixels from analysis")
            cube_flat = cube.reshape(ny*nx, -1)
            cube_flat[~valid_mask] = 0
            cube = cube_flat.reshape(ny, nx, -1)
    
    # 4.5. Spatial cluster masking (focus analysis on specific clusters)
    cluster_masking_result = None
    if cluster_mask is not None:
        logger.info(f"Applying spatial cluster mask: keeping clusters {cluster_mask}")
        
        # Run initial clustering if not already done
        if cluster_validation_results is None:
            logger.info("Running initial clustering for spatial masking...")
            
            temp_map = HyperspectralMap(
                cube=cube.copy(),
                energy=energy,
                metadata=map_data.metadata,
                name=map_data.name
            )
            
            initial_cluster = pca_cluster_analysis(
                temp_map,
                n_pca=min(3, n_pca),
                n_clusters=n_clusters,
                normalize='l1' if normalize_spectra else 'l2'
            )
            cluster_labels = initial_cluster['labels']
            silhouette = None
        else:
            cluster_labels = cluster_validation_results['cluster_results']['labels']
            silhouette = cluster_validation_results['validation_info'].get('silhouette_score')
        
        # Apply cluster mask using apply_mask module
        cluster_masking_result = apply_cluster_mask(
            cube=cube,
            energy=energy,
            cluster_labels=cluster_labels,
            focus_clusters=cluster_mask,
            n_clusters=n_clusters,
            silhouette=silhouette
        )
        
        # Apply mask to cube
        cube = cluster_masking_result.apply_to_cube(cube)
        
        # Save cluster mask visualization
        if make_plots:
            save_mask_visualization(
                cluster_masking_result.mask,
                (ny, nx),
                output_dir,
                map_data.name,
                'cluster_spatial',
                show=show_plots
            )
    
    # 5. Auto-decision for MCR based on heterogeneity indicators
    auto_mcr_triggered = False
    if not do_mcr and variability_metrics and pre_image is not None:
        # Decision rules:
        # 1. High spectral variability: mean_std > 15 OR max_std > 30
        # 2. High PRE heterogeneity: std(PRE) > 0.5
        mean_var = variability_metrics['mean_std']
        max_var = variability_metrics['max_std']
        pre_std = np.std(pre_image)
        
        high_variability = mean_var > 15 or max_var > 30
        high_pre_heterogeneity = pre_std > 0.5
        
        if high_variability or high_pre_heterogeneity:
            auto_mcr_triggered = True
            logger.info(f"Auto-triggering MCR analysis:")
            if high_variability:
                logger.info(f"  → High spectral variability (mean σ={mean_var:.1f}, max σ={max_var:.1f})")
            if high_pre_heterogeneity:
                logger.info(f"  → High PRE heterogeneity (σ={pre_std:.3f})")
    
    # 6. MCR-ALS analysis (non-negative decomposition)
    if do_mcr or auto_mcr_triggered:
        mode_str = "auto-triggered" if auto_mcr_triggered else "requested"
        logger.info(f"Running MCR analysis ({mode_str}) with auto component selection...")
        mcr_results = run_mcr_with_pca_init(
            cube, 
            n_components=None,  # Let PCA determine
            auto_select=True,
            variance_threshold=0.95,  # Increased from 0.90 to keep more components
            max_components=10,
            normalize=normalize_spectra
        )
        if mcr_results:
            logger.info(f"MCR method: {mcr_results['method']}")
            if mcr_results.get('auto_selected', False):
                pca_var = mcr_results.get('pca_variance', [])
                logger.info(f"PCA-guided selection: {mcr_results['n_components']} components explain {sum(pca_var)*100:.1f}% variance")
            
            # Visualize MCR components
            if make_plots:
                # Scree plot first (PCA component selection rationale)
                if mcr_results.get('auto_selected', False) and mcr_results.get('pca_variance') is not None:
                    from chemometrics_plots import plot_scree
                    plot_scree(
                        mcr_results['pca_variance'],
                        mcr_results['n_components'],
                        0.90,
                        output_dir,
                        base_name=map_data.name,
                        region=map_data.metadata.region or "",
                        show=show_plots
                    )
                
                # MCR component spectra and maps
                plot_mcr_components(
                    energy,
                    mcr_results['component_spectra'],
                    mcr_results['conc_maps'],
                    mcr_results['method'],
                    output_dir,
                    base_name=map_data.name,
                    region=map_data.metadata.region or "",
                    x_axis=map_data.x_axis,
                    y_axis=map_data.y_axis,
                    show=show_plots
                )
                
                # Compute and visualize MCR quality metrics
                from chemometrics_utils import compute_mcr_quality_metrics
                from chemometrics_plots import plot_mcr_quality_metrics
                
                quality_metrics = compute_mcr_quality_metrics(cube, mcr_results)
                logger.info(f"MCR quality: R²={quality_metrics['total_r2']:.3f}, "
                           f"LOF={quality_metrics['mean_lof']:.4f}")
                
                plot_mcr_quality_metrics(
                    quality_metrics,
                    output_dir,
                    base_name=map_data.name,
                    region=map_data.metadata.region or "",
                    x_axis=map_data.x_axis,
                    y_axis=map_data.y_axis,
                    show=show_plots
                )
                
                # Fit MCR components with peak fitting for chemical identification
                region = map_data.metadata.region or ""
                fitted_results = fit_mcr_components(
                    mcr_results=mcr_results,
                    energy=energy,
                    region=region,
                    template_dir=TEMPLATE_DIR,
                    output_dir=output_dir,
                    base_name=map_data.name
                )
                
                if fitted_results:
                    # Store fitted results in mcr_results
                    mcr_results['fitted_components'] = fitted_results
                    mcr_results['component_ids'] = fitted_results['component_labels']
                    logger.info(f"\n✓ Successfully fitted {len(fitted_results['component_fits'])} components")
                else:
                    # Fallback to simple peak matching if fitting fails
                    logger.info("Falling back to simple peak position matching...")
                    from chemometrics_utils import assign_chemical_states
                    component_ids = assign_chemical_states(
                        mcr_results['component_spectra'],
                        energy,
                        region or "Unknown"
                    )
                    mcr_results['component_ids'] = component_ids
                    logger.info("MCR component assignments (simple matching):")
                    for i, label in enumerate(component_ids):
                        logger.info(f"  Component {i}: {label}")
    
    # 6. Update map_data with processed cube
    processed_map = HyperspectralMap(
        cube=cube,
        energy=energy,
        metadata=map_data.metadata,
        name=map_data.name
    )
    
    # 7. Run PCA and clustering analysis
    cluster_results = pca_cluster_preselect(
        processed_map,
        n_pca=min(n_pca, min(ny*nx, len(energy))-1),
        n_clusters=min(n_clusters, ny*nx),
        normalize='l1' if normalize_spectra else 'l2',
        output_dir=output_dir
    )
    
    # 8. Plot cluster analysis
    if make_plots:
        logger.info("Generating cluster analysis plots...")
        # Use processed_map (with cropped energy) instead of original map_data
        plot_cluster_analysis(processed_map, cluster_results, output_dir, show_plots, base_name=map_data.name)
        
        # Also plot cluster spectra
        plot_cluster_spectra(
            energy, 
            cluster_results['cluster_info'],
            output_dir,
            base_name=map_data.name + "_cluster_spectra",
            plot_type="overlay"
        )
        
        # Plot cluster size distribution
        plot_cluster_size_distribution(
            cluster_results['cluster_info'],
            output_dir,
            base_name=map_data.name + "_cluster_sizes"
        )
    
    # 9. Extract representative spectra from clusters
    component_maps = {}
    for cluster_info in cluster_results['cluster_info']:
        if cluster_info['size'] > 0 and cluster_info['mean_spec'] is not None:
            cluster_id = cluster_info['cluster']
            # Create a map showing this cluster's pixels
            cluster_map = (cluster_results['labels'] == cluster_id).astype(float)
            component_maps[f"Cluster_{cluster_id}"] = cluster_map * cluster_info['size']
    
    # 10. Add MCR component maps if available
    if mcr_results:
        for i in range(mcr_results['conc_maps'].shape[2]):
            component_maps[f"MCR_Component_{i}"] = mcr_results['conc_maps'][:, :, i]
    
    return {
        'cluster_results': cluster_results,
        'component_maps': component_maps,
        'pca_results': cluster_results['pca'],
        'pre_image': pre_image,
        'mask': mask,
        'mcr_results': mcr_results,
        'cluster_validation': cluster_validation_results,
        'cluster_masking': cluster_masking_result,
        'variability_metrics': variability_metrics
    }

# ============================ Baseline functions (fallback if import fails) ============================
def shirley_baseline_local(E: np.ndarray, I: np.ndarray, 
                         tol: float = 1e-5, max_iter: int = 50) -> np.ndarray:
    """
    Calculate Shirley background using iterative method.
    Fallback implementation if import from XPS_peakfitting_V2 fails.
    
    Args:
        E: Binding energy array (must be ascending)
        I: Intensity array
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Background array same length as input arrays
    """
    # Ensure ascending energy
    if E[0] > E[-1]:
        E = E[::-1]
        I = I[::-1]
        
    # Initial guess: straight line
    bg = np.linspace(I[0], I[-1], len(I))
    
    # Iterative procedure
    for _ in range(max_iter):
        bg_old = bg.copy()
        # Integration factor at each point
        integral = np.array([np.trapz(I[i:] - bg[i:], E[i:]) 
                           for i in range(len(E))])
        # New background estimate
        bg = I[0] + (I[-1] - I[0]) * integral / integral[0]
        # Check convergence
        if np.max(np.abs(bg - bg_old)) < tol:
            break
            
    return bg

# plot_cluster_analysis now imported from cluster_plots module

# ============================ Save results ============================
# save_mcr_fitting_results now imported from mcr_fitting module


# save_quantitative_concentration_maps now imported from mcr_fitting module


def save_2d_results(parsed: Map2D, denoised: np.ndarray, mask: np.ndarray, stats: Dict, file_path: Path, output_dir: Path):
    base_name = file_path.stem
    region = parsed.metadata.region or "unknown"
    
    # Count total spectra (pixels)
    total_spectra = np.prod(parsed.shape)
    valid_spectra = np.sum(mask)  # Count pixels in ROI
    
    # CSVs / text
    np.savetxt(output_dir / f"{base_name}_{region}_denoised.csv", denoised, delimiter=',', fmt='%.6f')
    np.savetxt(output_dir / f"{base_name}_{region}_mask.csv", mask.astype(int), delimiter=',', fmt='%d')
    
    with open(output_dir / f"{base_name}_{region}_stats.txt", 'w') as f:
        f.write("XPS 2D Map Analysis Results\n")
        f.write(f"File: {file_path.name}\n")
        f.write(f"Region: {region}\n")
        f.write(f"Shape: {parsed.shape}\n")
        f.write(f"Total Spectra: {total_spectra}\n")
        f.write(f"Valid Spectra in ROI: {valid_spectra} ({valid_spectra/total_spectra*100:.1f}%)\n")
        f.write("\nROI Statistics:\n")
        for key, value in stats.items():
            f.write(f" {key}: {value}\n")
    logger.info(f"Saved 2D analysis to {output_dir}")

def save_hyperspectral_results(parsed: HyperspectralMap, results: Dict, file_path: Path, output_dir: Path):
    base_name = file_path.stem
    region = parsed.metadata.region or "unknown"
    
    # Count spectra statistics
    ny, nx, nE = parsed.shape
    total_spectra = ny * nx
    valid_spectra = np.sum(np.any(parsed.cube != 0, axis=2))  # Count non-zero spectra
    
    # Handle both old format (fitting-based) and new format (clustering-based)
    component_maps = results.get('component_maps', {})
    r2_map = results.get('r2_map', np.zeros((ny, nx)))
    fitted_spectra = np.sum(r2_map > 0) if 'r2_map' in results else 0
    
    # Extract cluster_info and pca_info from nested structure if needed
    cluster_results = results.get('cluster_results', {})
    cluster_info = results.get('cluster_info', cluster_results.get('cluster_info', []))
    pca_results = results.get('pca_results', results.get('pca', {}))
    
    # Save component maps (cluster maps)
    for comp_name, comp_map in component_maps.items():
        np.savetxt(output_dir / f"{base_name}_{region}_{comp_name}.csv", comp_map, delimiter=',', fmt='%.6f')
    
    # Save cluster labels if available
    if cluster_results and 'labels' in cluster_results:
        np.savetxt(output_dir / f"{base_name}_{region}_cluster_labels.csv", 
                  cluster_results['labels'], delimiter=',', fmt='%d')
    
    # Save PCA score maps if available
    if pca_results and 'score_maps' in pca_results:
        score_maps = pca_results['score_maps']
        for i in range(score_maps.shape[2]):
            np.savetxt(output_dir / f"{base_name}_{region}_PC{i+1}_scores.csv", 
                      score_maps[:,:,i], delimiter=',', fmt='%.6f')
    
    # Save PRE image if available
    if 'pre_image' in results and results['pre_image'] is not None:
        np.savetxt(output_dir / f"{base_name}_{region}_PRE.csv", 
                  results['pre_image'], delimiter=',', fmt='%.6f')
    
    # Save mask if available
    if 'mask' in results and results['mask'] is not None:
        np.savetxt(output_dir / f"{base_name}_{region}_mask.csv", 
                  results['mask'].astype(int), delimiter=',', fmt='%d')
    
    # Save MCR results if available
    mcr_results = results.get('mcr_results', {})
    if mcr_results:
        # Save MCR component spectra
        component_spectra = mcr_results.get('component_spectra')
        if component_spectra is not None:
            for i in range(component_spectra.shape[1]):
                spectrum_data = np.column_stack([parsed.energy, component_spectra[:, i]])
                np.savetxt(output_dir / f"{base_name}_{region}_MCR_component{i}_spectrum.csv",
                          spectrum_data, delimiter=',',
                          header='Energy(eV),Intensity', comments='', fmt='%.6f')
        
        # Save detailed MCR fitting results if available
        fitted_components = mcr_results.get('fitted_components')
        if fitted_components:
            # Export peak fitting parameters to CSV
            save_mcr_fitting_results(mcr_results, output_dir, base_name, region, parsed.energy)
            
            # Export quantitatively scaled concentration maps
            save_quantitative_concentration_maps(mcr_results, output_dir, base_name, region, ny, nx)
            
            # Generate MCR fitting visualization plots
            
            plots_output = output_dir.parent.parent / "plots" / "hyperspectral_maps"
            plots_output.mkdir(parents=True, exist_ok=True)
            
            try:
                logger.info(f"Generating MCR fitting plots in {plots_output}...")
                
                # 1. Fitted component spectra with peak deconvolution
                plot_mcr_fitting_results(
                    mcr_results, parsed.energy, plots_output,
                    base_name, region, show=False
                )
                
                # 2. Atomic percentage bar chart and pie chart
                plot_atomic_percentages(
                    fitted_components, plots_output,
                    base_name, region, 
                    method=mcr_results.get("method", "MCR"),
                    show=False
                )
                
                # 3. Quantitative concentration maps
                plot_quantitative_concentration_maps(
                    mcr_results, plots_output, base_name, region,
                    x_axis=parsed.x_axis,
                    y_axis=parsed.y_axis,
                    show=False
                )
                
                # 4. Peak parameter summary table
                plot_peak_parameter_summary(
                    fitted_components, plots_output,
                    base_name, region, show=False
                )
                
                # 5. Combined concentration map (all species side-by-side)
                plot_combined_concentration_maps(
                    mcr_results, plots_output, base_name, region,
                    x_axis=parsed.x_axis,
                    y_axis=parsed.y_axis,
                    show=False
                )
                
                logger.info(f"✓ Generated MCR fitting visualization plots for {region}")
            except Exception as e:
                logger.error(f"Failed to generate MCR fitting plots: {e}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
    
    # Save R² map if from fitting-based workflow
    if 'r2_map' in results:
        np.savetxt(output_dir / f"{base_name}_{region}_r2_map.csv", r2_map, delimiter=',', fmt='%.6f')
    
    # Save average spectrum
    if 'avg_spectrum' in results:
        spectrum_data = np.column_stack([parsed.energy, results['avg_spectrum']])
        np.savetxt(output_dir / f"{base_name}_{region}_avg_spectrum.csv", 
                  spectrum_data, delimiter=',', 
                  header='Energy(eV),Intensity', comments='', fmt='%.6f')

    with open(output_dir / f"{base_name}_{region}_summary.txt", 'w') as f:
        f.write("XPS Hyperspectral Map Analysis Results\n")
        f.write(f"File: {file_path.name}\n")
        f.write(f"Region: {region}\n")
        f.write(f"Shape: {parsed.shape}\n")
        f.write(f"Energy range: {parsed.energy[0]:.2f} to {parsed.energy[-1]:.2f} eV\n")
        
        f.write("\nSpectra Statistics:\n")
        f.write(f"Total pixels: {total_spectra}\n")
        f.write(f"Valid spectra: {valid_spectra} ({valid_spectra/total_spectra*100:.1f}%)\n")
        if fitted_spectra > 0:
            f.write(f"Successfully fitted: {fitted_spectra} ({fitted_spectra/total_spectra*100:.1f}%)\n")
        
        # Cluster statistics
        if cluster_info:
            f.write("\nCluster Statistics:\n")
            for cluster in cluster_info:
                if cluster['size'] > 0:
                    f.write(f"Cluster {cluster['cluster']}: {cluster['size']} spectra ")
                    f.write(f"({cluster['size']/total_spectra*100:.1f}%)\n")
        
        # PCA statistics
        if pca_results and 'explained_variance' in pca_results:
            f.write("\nPCA Analysis:\n")
            explained_var = pca_results['explained_variance']
            total_var = np.sum(explained_var)
            for i, var in enumerate(explained_var):
                f.write(f"PC{i+1}: {var*100:.1f}% variance explained\n")
            f.write(f"Total variance explained: {total_var*100:.1f}%\n")
        
        # PRE statistics
        if 'pre_image' in results and results['pre_image'] is not None:
            pre_img = results['pre_image']
            f.write("\nPattern Recognition Entropy (PRE):\n")
            f.write(f"Mean: {np.mean(pre_img):.3f}\n")
            f.write(f"Std: {np.std(pre_img):.3f}\n")
            f.write(f"Range: [{np.min(pre_img):.3f}, {np.max(pre_img):.3f}]\n")
        
        # MCR statistics
        mcr_results = results.get('mcr_results', {})
        if mcr_results:
            f.write(f"\nMCR Analysis:\n")
            f.write(f"Method: {mcr_results.get('method', 'Unknown')}\n")
            n_comp = mcr_results.get('component_spectra', np.array([])).shape[1] if 'component_spectra' in mcr_results else 0
            f.write(f"Number of components: {n_comp}\n")
        
        # Mask statistics
        if 'mask' in results and results['mask'] is not None:
            mask_data = results['mask']
            masked_pixels = np.sum(mask_data)
            f.write("\nMasking Statistics:\n")
            f.write(f"Valid pixels: {masked_pixels} ({masked_pixels/total_spectra*100:.1f}%)\n")
        
        # Component statistics
        if component_maps:
            f.write("\nCluster Map Areas:\n")
            for comp_name, comp_map in component_maps.items():
                nonzero_mask = comp_map > 0
                if np.any(nonzero_mask):
                    mean_area = np.mean(comp_map[nonzero_mask])
                    std_area = np.std(comp_map[nonzero_mask])
                    n_pixels = np.sum(nonzero_mask)
                    f.write(f"{comp_name}: {mean_area:.2f} ± {std_area:.2f} ")
                    f.write(f"(in {n_pixels} pixels, {n_pixels/total_spectra*100:.1f}%)\n")
    logger.info(f"Saved hyperspectral analysis to {output_dir}")


# ============================ Main orchestration (minimal prints) ============================
def main(make_plots: bool = True, show_plots: bool = False, verbose: bool = False):
    setup_logger(verbose=verbose)
    start_time_total = time.time()

    # Setup directories
    ensure_output_directories()
    logger.info(f"Map processor initialized")
    logger.info(f"Input directory: {RAW_DATA_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Load configuration
    try:
        config, region_defs = load_xps_config(CONFIG_FILE)
        logger.info(f"Configuration loaded from: {CONFIG_FILE}")
    except Exception as e:
        logger.info(f"Config load failed: {e}. Proceeding with defaults.")
        config = {}
        region_defs = {}

    # Check data directory
    if not RAW_DATA_DIR.exists():
        logger.error(f"Data directory not found: {RAW_DATA_DIR}")
        logger.info(f"Please ensure map data is available in the raw data directory.")
        return

    # Find map files
    logger.info(f"Scanning for map files in: {RAW_DATA_DIR}")
    try:
        map_files = find_map_files(str(RAW_DATA_DIR), RAW_FILE_PATTERNS)
    except Exception as e:
        logger.error(f"Failed to scan data directory: {e}")
        return

    if not map_files:
        logger.warning(f"No map files found (patterns: {RAW_FILE_PATTERNS})")
        logger.info(f"Supported formats: {', '.join(RAW_FILE_PATTERNS)}")
        return
        
    logger.info(f"Found {len(map_files)} map file(s) for processing")

    # Process each file
    results_summary = []
    file_timings = []

    for idx, file_path in enumerate(map_files, 1):
        start_time_file = time.time()
        fp = Path(file_path)
        logger.info(f"[{idx}/{len(map_files)}] Processing: {fp.name}")

        try:
            # Log spectra summary before processing
            _ = log_map_spectra_summary(str(fp))
            
            # Process the file
            parsed = parse_map_with_config(str(fp), override_nx=None, override_ny=None)
            if isinstance(parsed, Map2D):
                logger.info(f"Type: 2D map | Region: {parsed.metadata.region} | Shape: {parsed.shape}")
                analysis_dir = OUTPUT_DIR / "analysis_results" / "2d_maps"
                plots_dir = OUTPUT_DIR / "plots" / "2d_maps"
                outputs = process_2d_map(parsed, fp, analysis_dir, plots_dir, make_plots=make_plots, show_plots=show_plots)
                
                # Generate analysis report for 2D maps
                try:
                    # Prepare metadata
                    ny, nx = parsed.shape
                    hmap_metadata = {
                        "region": parsed.metadata.region or "Unknown",
                        "file": parsed.metadata.source_file or "N/A",
                        "format": parsed.metadata.source_format or "Unknown",
                        "nx": nx,
                        "ny": ny,
                        "total_pixels": nx * ny,
                        "n_energy_points": 1
                    }
                    
                    # Scan for plot files in 2d_maps directory (only include plots for this dataset)
                    plot_paths = {}
                    plots_scan_dir = OUTPUT_DIR / "plots" / "2d_maps"
                    base_name = fp.stem.lower()
                    if plots_scan_dir.exists():
                        import re
                        for plot_file in sorted(plots_scan_dir.iterdir()):
                            if not plot_file.is_file() or not plot_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                                continue
                            stem = plot_file.stem.lower()
                            # Only include files matching this dataset's base name
                            if base_name not in stem:
                                continue
                            # Map 2D plot filenames to expected keys
                            if '2d_overview' in stem or 'overview' in stem:
                                plot_paths.setdefault('overview', plot_file)
                    
                    # Generate HTML report
                    report_path_html = generate_comprehensive_report(
                        output_dir=OUTPUT_DIR,
                        hmap_metadata=hmap_metadata,
                        pca_results=None,
                        mcr_results=None,
                        cluster_results=None,
                        nnls_results=None,
                        pixel_diagnostics=None,
                        plot_paths=plot_paths,
                        format="html"
                    )
                    if report_path_html:
                        logger.info(f"  ✓ HTML report: {report_path_html.name}")
                    
                    # Generate PDF report
                    try:
                        report_path_pdf = generate_comprehensive_report(
                            output_dir=OUTPUT_DIR,
                            hmap_metadata=hmap_metadata,
                            pca_results=None,
                            mcr_results=None,
                            cluster_results=None,
                            nnls_results=None,
                            pixel_diagnostics=None,
                            plot_paths=plot_paths,
                            format="pdf"
                        )
                        if report_path_pdf:
                            logger.info(f"  ✓ PDF report: {report_path_pdf.name}")
                    except ImportError as pdf_err:
                        logger.warning(f"  ⚠ PDF generation skipped (WeasyPrint not installed)")
                    except Exception as pdf_err:
                        logger.warning(f"  ⚠ PDF generation failed: {pdf_err}")
                except Exception as report_err:
                    logger.warning(f"  ✗ Report generation failed: {report_err}")
                
                results_summary.append({
                    'file': fp.name, 'type': '2D map', 'region': parsed.metadata.region,
                    'shape': parsed.shape, 'status': 'success'
                })
            elif isinstance(parsed, HyperspectralMap):
                logger.info(f"Type: Hyperspectral | Region: {parsed.metadata.region} | Shape: {parsed.shape}")
                
                # Create analysis and plots subdirectories
                analysis_dir = OUTPUT_DIR / "analysis_results" / "hyperspectral_maps"
                plots_dir = OUTPUT_DIR / "plots" / "hyperspectral_maps"
                analysis_dir.mkdir(parents=True, exist_ok=True)
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                outputs = process_hyperspectral_map_simple(
                    map_data=parsed,
                    output_dir=plots_dir,
                    config=config,  # Pass config for element-specific ranges
                    make_plots=make_plots,
                    show_plots=show_plots,
                    n_pca=3,
                    n_clusters=4,
                    do_nmf=True,
                    n_nmf=3,
                    compute_pre=True,
                    charge_ref_energy=284.8,  # Set to 284.8 for C1s charge correction
                    mask_threshold=2.0,      # Set to filter low-count pixels
                    normalize_spectra=False,
                    do_mcr=MCR_AVAILABLE,     # Enable MCR if pymcr available
                    n_mcr=3
                )
                
                # Save hyperspectral analysis results
                save_hyperspectral_results(parsed, outputs, fp, analysis_dir)
                
                # Generate analysis report
                try:
                    # Prepare metadata
                    ny, nx, nE = parsed.shape
                    hmap_metadata = {
                        "region": parsed.metadata.region or "Unknown",
                        "file": parsed.metadata.source_file or "N/A",
                        "format": parsed.metadata.source_format or "Unknown",
                        "nx": nx,
                        "ny": ny,
                        "total_pixels": nx * ny,
                        "n_energy_points": nE,
                        "energy_min": float(parsed.energy.min()),
                        "energy_max": float(parsed.energy.max())
                    }
                    
                    # Extract PCA results
                    pca_results = outputs.get('pca_results')
                    
                    # Extract MCR results
                    mcr_results = outputs.get('mcr_results')
                    
                    # Extract cluster results
                    cluster_results = None
                    if 'cluster_results' in outputs and outputs['cluster_results']:
                        cluster_data = outputs['cluster_results']
                        cluster_results = {
                            'n_clusters': len(cluster_data.get('cluster_info', [])),
                            'cluster_info': cluster_data.get('cluster_info', []),
                            'silhouette_score': cluster_data.get('silhouette_score', 0.0)
                        }
                    
                    # Pixel diagnostics
                    pixel_diagnostics = None
                    if 'mse_map' in outputs:
                        mse_map = outputs['mse_map']
                        pixel_diagnostics = {
                            'mean_mse': float(np.mean(mse_map)),
                            'problematic_pixels': int(np.sum(mse_map > 3 * np.median(mse_map)))
                        }
                    
                    # Scan for plot files in hyperspectral_maps directory (only include plots for this dataset)
                    plot_paths = {}
                    plots_scan_dir = OUTPUT_DIR / "plots" / "hyperspectral_maps"
                    base_name = fp.stem.lower()
                    if plots_scan_dir.exists():
                        import re
                        for plot_file in sorted(plots_scan_dir.iterdir()):
                            if not plot_file.is_file() or not plot_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                                continue
                            stem = plot_file.stem.lower()
                            # Include only files that have this dataset's base name
                            if base_name not in stem:
                                continue
                            # PCA scree
                            if ('scree' in stem and 'pca' in stem) or stem.endswith('_scree_plot'):
                                plot_paths.setdefault('pca_scree', plot_file)
                            # PCA general plot / loadings
                            if ('_pca' in stem) or ('pca_analysis' in stem) or (stem == 'pca'):
                                plot_paths.setdefault('pca_loadings', plot_file)
                            # PC score maps
                            m = re.search(r'pc(?:_|)(\d+).*(score|scores|scores?_map|_scores)', stem)
                            if m:
                                idx = int(m.group(1)) - 1
                                plot_paths.setdefault(f'pca_score_map_{idx}', plot_file)
                                continue
                            m2 = re.search(r'pc(\d+).*score', stem)
                            if m2:
                                idx = int(m2.group(1)) - 1
                                plot_paths.setdefault(f'pca_score_map_{idx}', plot_file)
                                continue
                            # MCR components and quality
                            if 'mcr_components' in stem or 'mcr_component' in stem:
                                plot_paths.setdefault('mcr_spectra', plot_file)
                            if 'mcr_quality' in stem:
                                plot_paths.setdefault('mcr_quality', plot_file)
                            # Cluster maps and spectra (prioritize cluster_analysis)
                            if 'cluster_analysis' in stem:
                                plot_paths['cluster_map'] = plot_file
                            elif 'cluster_sizes' in stem and 'cluster_map' not in plot_paths:
                                plot_paths.setdefault('cluster_map', plot_file)
                            if 'cluster_spectra' in stem or 'cluster_spectrum' in stem:
                                plot_paths.setdefault('cluster_spectra', plot_file)
                            # Combined concentration map and individual species maps
                            if 'combined_concentration_map' in stem:
                                plot_paths.setdefault('combined_concentration_map', plot_file)
                            if 'individual_species_maps' in stem:
                                plot_paths.setdefault('individual_species_maps', plot_file)
                            # PRE map and spectra overlay
                            if 'pre_map' in stem:
                                plot_paths.setdefault('pre_map', plot_file)
                            if 'spectra_overlay' in stem:
                                plot_paths.setdefault('spectra_overlay', plot_file)
                            # Diagnostics
                            if 'mse_map' in stem:
                                plot_paths.setdefault('mse_map', plot_file)
                            if 'shift_map' in stem:
                                plot_paths.setdefault('shift_map', plot_file)
                    
                    logger.info(f"Scanned {len(plot_paths)} plot files for report")
                    
                    # Generate HTML report
                    report_path_html = generate_comprehensive_report(
                        output_dir=OUTPUT_DIR,
                        hmap_metadata=hmap_metadata,
                        pca_results=pca_results,
                        mcr_results=mcr_results,
                        cluster_results=cluster_results,
                        nnls_results=None,
                        pixel_diagnostics=pixel_diagnostics,
                        plot_paths=plot_paths,
                        format="html"
                    )
                    if report_path_html:
                        logger.info(f"  ✓ HTML report: {report_path_html.name}")
                    
                    # Generate PDF report
                    try:
                        report_path_pdf = generate_comprehensive_report(
                            output_dir=OUTPUT_DIR,
                            hmap_metadata=hmap_metadata,
                            pca_results=pca_results,
                            mcr_results=mcr_results,
                            cluster_results=cluster_results,
                            nnls_results=None,
                            pixel_diagnostics=pixel_diagnostics,
                            plot_paths=plot_paths,
                            format="pdf"
                        )
                        if report_path_pdf:
                            logger.info(f"  ✓ PDF report: {report_path_pdf.name}")
                    except ImportError as pdf_err:
                        logger.warning(f"  ⚠ PDF generation skipped (WeasyPrint not installed)")
                    except Exception as pdf_err:
                        logger.warning(f"  ⚠ PDF generation failed: {pdf_err}")
                except Exception as report_err:
                    logger.warning(f"  ✗ Report generation failed: {report_err}")
                
                results_summary.append({
                    'file': fp.name, 'type': 'Hyperspectral', 'region': parsed.metadata.region,
                    'shape': parsed.shape, 'status': 'success'
                })
            
            elapsed_file = time.time() - start_time_file
            file_timings.append(elapsed_file)
            logger.info(f"Done {fp.name} ({format_time(elapsed_file)})")
        except Exception as e:
            elapsed_file = time.time() - start_time_file
            file_timings.append(elapsed_file)
            logger.info(f"Error processing {fp.name}: {e}")
            if verbose:
                traceback.print_exc()
            results_summary.append({
                'file': fp.name, 'type': 'Unknown', 'region': 'N/A', 'shape': 'N/A', 'status': f'failed: {str(e)}'
            })

    # Summary
    elapsed_total = time.time() - start_time_total
    n_success = sum(1 for r in results_summary if r['status'] == 'success')
    n_failed = len(results_summary) - n_success
    logger.info("Summary:")
    logger.info(f"  Files processed: {len(map_files)} | Success: {n_success} | Failed: {n_failed}")
    logger.info(f"  Total time: {format_time(elapsed_total)}")
    if file_timings:
        avg_time = sum(file_timings) / len(file_timings)
        logger.info(f"  Avg per file: {format_time(avg_time)}")

# ============================ Entry point ============================
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Process XPS map files (2D single-energy or hyperspectral).")
    parser.add_argument("file", nargs="?", type=str, default=None, help="Path to XPS map ASCII file (optional, if omitted batch mode)")
    parser.add_argument("--nx", type=int, default=None, help="Override nx (pixels along X)")
    parser.add_argument("--ny", type=int, default=None, help="Override ny (pixels along Y)")
    parser.add_argument("--off", type=str, default=None, help="Optional off-peak map file (Case 1 only)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian denoise sigma (Case 1)")
    parser.add_argument("--threshold", type=str, default="otsu", choices=["otsu", "percentile"], help="Thresholding method (Case 1)")
    parser.add_argument("--percentile", type=float, default=95.0, help="Percentile for thresholding if method=percentile")
    parser.add_argument("--morph", type=str, default="open", choices=["open", "close"], help="Morph operation (Case 1)")
    parser.add_argument("--morph_size", type=int, default=2, help="Morph kernel size (Case 1)")
    parser.add_argument("--init_peaks", type=str, default=None, help="Comma-separated peak centers:sigmas for hyperspectral (e.g., '284.8:0.3,285.3:0.3')")
    parser.add_argument("--bg_lam", type=float, default=1e5, help="ALS baseline lambda (Hyperspectral)")
    parser.add_argument("--bg_p", type=float, default=0.01, help="ALS baseline asymmetry p (Hyperspectral)")
    parser.add_argument("--bg_niter", type=int, default=10, help="ALS baseline iterations (Hyperspectral)")
    parser.add_argument("--max_shift_bins", type=int, default=10, help="Max bins for shift estimation (Hyperspectral)")
    parser.add_argument("--no_pca", action="store_true", help="Disable PCA (Hyperspectral)")
    parser.add_argument("--n_pca", type=int, default=3, help="Number of PCA components")
    parser.add_argument("--no_nmf", action="store_true", help="Disable NMF (Hyperspectral)")
    parser.add_argument("--n_nmf", type=int, default=3, help="Number of NMF components")
    parser.add_argument("--no_plots", action="store_true", help="Disable saving plots")
    parser.add_argument("--show_plots", action="store_true", help="Show plots interactively")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--pca_preselect", action="store_true", help="Use PCA+clustering to preselect reps and NNLS propagate")
    parser.add_argument("--n_clusters", type=int, default=4, help="Number of clusters in PCA score space")
    parser.add_argument("--pca_components", type=int, default=3, help="Number of PCA components")
    parser.add_argument("--rep_mode", type=str, default="mean", choices=["mean", "medoid"], help="Representative spectrum per cluster")
    parser.add_argument("--clusters_to_fit", type=str, default=None, help="Comma-separated cluster IDs to fit (e.g., '0,2')")
    parser.add_argument("--template_name", type=str, default=None, help="Template YAML for external fitter")
    parser.add_argument("--template_dir", type=str, default=None, help="Directory for templates")
    args = parser.parse_args()
    setup_logger(verbose=args.verbose)

    make_plots = not args.no_plots
    show_plots = args.show_plots

    if args.file:
        # Advanced single-file mode
        logger.info(f"Processing {args.file}")
        init_peaks = None
        if args.init_peaks:
            pairs = [p.strip() for p in args.init_peaks.split(",") if p.strip()]
            init_peaks = []
            for ps in pairs:
                try:
                    c, s = [float(x) for x in ps.split(":")]
                    init_peaks.append((c, s))
                except Exception:
                    pass
        try:
            outputs = process_xps_map(
                args.file,
                off_peak_path=args.off,
                case1_sigma=args.sigma,
                case1_threshold_method=args.threshold,
                case1_percentile=args.percentile,
                case1_morph_op=args.morph,
                case1_morph_size=args.morph_size,
                init_peaks=init_peaks,
                bg_lam=args.bg_lam,
                bg_p=args.bg_p,
                bg_niter=args.bg_niter,
                max_shift_bins=args.max_shift_bins,
                do_pca=not args.no_pca,
                n_pca=args.n_pca,
                do_nmf=not args.no_nmf,
                n_nmf=args.n_nmf,
                make_plots=make_plots,
                show_plots=show_plots,
                override_nx=args.nx,
                override_ny=args.ny
            )

            parsed = outputs["parsed"]
            if isinstance(parsed, Map2D):
                logger.info(f"Done 2D map: region={parsed.metadata.region}, shape={parsed.shape}")
            else:
                logger.info(f"Done hyperspectral: region={parsed.metadata.region}, shape={parsed.shape}, nE={parsed.energy.size}")
        except Exception as e:
            logger.info(f"Error: {e}")
            if args.verbose:
                traceback.print_exc()
    else:
        # Batch mode
        main(make_plots=make_plots, show_plots=show_plots, verbose=args.verbose)
