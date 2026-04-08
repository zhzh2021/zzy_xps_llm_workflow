"""
Example script demonstrating automated report generation for XPS map analysis.

This script shows how to:
1. Process a hyperspectral XPS map
2. Perform PCA, MCR-ALS, and clustering analysis
3. Generate a comprehensive HTML/PDF report

Usage:
    python test_report_generator.py --file path/to/map.txt --format html
    python test_report_generator.py --file path/to/map.txt --format pdf
"""

import sys
from pathlib import Path
import numpy as np
import argparse

# Add mapper directory to path
mapper_dir = Path(__file__).parent
if str(mapper_dir) not in sys.path:
    sys.path.insert(0, str(mapper_dir))

from XPS_map import (
    parse_map_with_config,
    HyperspectralMap,
    OUTPUT_DIR,
    setup_logger
)
from report_generator import (
    generate_comprehensive_report,
    create_pca_report_data,
    create_mcr_report_data,
    create_cluster_report_data
)
from case2_hyperspectral_processing import (
    process_hyperspectral,
    pca_cluster_analysis
)
from chemometrics_utils import run_mcr_on_cube, MCR_AVAILABLE
import logging

logger = logging.getLogger("xps_map")


def create_synthetic_hyperspectral_map(
    nx: int = 20,
    ny: int = 20,
    nE: int = 100,
    energy_range: tuple = (280.0, 295.0)
) -> HyperspectralMap:
    """
    Create a synthetic hyperspectral map for testing.
    
    Args:
        nx: Number of x pixels
        ny: Number of y pixels
        nE: Number of energy points
        energy_range: (E_min, E_max) in eV
        
    Returns:
        Synthetic HyperspectralMap
    """
    from map_parser import HyperspectralMap, MapMetadata
    
    # Generate energy axis
    energy = np.linspace(energy_range[0], energy_range[1], nE)
    
    # Create synthetic data cube with 3 spatial patterns
    cube = np.zeros((ny, nx, nE))
    
    # Pattern 1: Gaussian peak at 284.8 eV (C-C) - top-left quadrant
    E1 = 284.8
    sigma1 = 0.8
    for y in range(ny):
        for x in range(nx):
            weight = np.exp(-((x-nx/4)**2 + (y-ny/4)**2) / (nx/3)**2)
            cube[y, x, :] += weight * 1000 * np.exp(-0.5 * ((energy - E1) / sigma1)**2)
    
    # Pattern 2: Gaussian peak at 286.5 eV (C-O) - top-right quadrant
    E2 = 286.5
    sigma2 = 0.9
    for y in range(ny):
        for x in range(nx):
            weight = np.exp(-((x-3*nx/4)**2 + (y-ny/4)**2) / (nx/3)**2)
            cube[y, x, :] += weight * 800 * np.exp(-0.5 * ((energy - E2) / sigma2)**2)
    
    # Pattern 3: Gaussian peak at 288.2 eV (C=O) - bottom half
    E3 = 288.2
    sigma3 = 1.0
    for y in range(ny):
        for x in range(nx):
            weight = np.exp(-((y-3*ny/4)**2) / (ny/3)**2)
            cube[y, x, :] += weight * 600 * np.exp(-0.5 * ((energy - E3) / sigma3)**2)
    
    # Add noise
    cube += np.random.normal(0, 20, cube.shape)
    cube = np.maximum(cube, 0)  # No negative counts
    
    # Create metadata
    metadata = MapMetadata(
        region="C1s",
        x_start=0.0,
        x_step=1.0,
        nx=nx,
        y_start=0.0,
        y_step=1.0,
        ny=ny,
        energy_axis=energy,
        source_format="synthetic",
        source_file="synthetic_c1s_map.txt"
    )
    
    return HyperspectralMap(cube=cube, energy=energy, metadata=metadata)


def test_report_generation(
    hmap: HyperspectralMap,
    output_dir: Path,
    format: str = "html",
    run_mcr: bool = True
):
    """
    Test the full analysis and report generation pipeline.
    
    Args:
        hmap: HyperspectralMap to analyze
        output_dir: Output directory
        format: Report format ('html' or 'pdf')
        run_mcr: Whether to run MCR-ALS (requires pymcr)
    """
    logger.info(f"Testing report generation with {format.upper()} format")
    logger.info(f"Map shape: {hmap.shape}, Energy range: {hmap.energy.min():.1f}-{hmap.energy.max():.1f} eV")
    
    # Define initial peaks for fitting
    init_peaks = [
        (284.8, 0.8),  # C-C
        (286.5, 0.9),  # C-O
        (288.2, 1.0)   # C=O
    ]
    
    # Simplified processing - just do PCA and NMF without pixel fitting
    logger.info("Running PCA and NMF analysis...")
    ny, nx, nE = hmap.shape
    
    # Initialize outputs dict
    outputs = {}
    
    # PCA
    from sklearn.decomposition import PCA, NMF
    X = hmap.cube.reshape(ny*nx, nE)
    pca = PCA(n_components=3, whiten=False, random_state=0)
    scores = pca.fit_transform(X)
    outputs["pca"] = {
        "components": pca.components_,
        "score_maps": scores.reshape(ny, nx, 3),
        "explained_variance": pca.explained_variance_ratio_
    }
    
    # NMF
    X_nonneg = np.maximum(X, 0)
    nmf = NMF(n_components=3, init="nndsvd", random_state=0, max_iter=500)
    W = nmf.fit_transform(X_nonneg)
    outputs["nmf"] = {
        "components": nmf.components_,
        "abundance_maps": W.reshape(ny, nx, 3),
        "reconstruction_error": nmf.reconstruction_err_
    }
    
    # Create dummy diagnostic maps
    outputs["mse_map"] = np.random.rand(ny, nx) * 0.1 + 0.05
    outputs["shift_map"] = np.random.randn(ny, nx) * 0.2
    
    # Add PCA clustering
    logger.info("Running PCA clustering analysis...")
    cluster_outputs = pca_cluster_analysis(
        hmap=hmap,
        n_pca=3,
        n_clusters=4,
        use_minibatch=False,
        normalize="l2"
    )
    
    outputs["cluster_labels"] = cluster_outputs["labels"]
    outputs["cluster_info"] = cluster_outputs["cluster_info"]
    outputs["n_clusters"] = 4
    
    # Calculate silhouette score
    from sklearn.metrics import silhouette_score
    scores = cluster_outputs["pca"]["score_maps"].reshape(-1, 3)
    labels = cluster_outputs["labels"].reshape(-1)
    outputs["silhouette_score"] = silhouette_score(scores, labels)
    
    # Run MCR-ALS if available
    if run_mcr and MCR_AVAILABLE:
        logger.info("Running MCR-ALS...")
        try:
            ny, nx, nE = hmap.shape
            mcr_result = run_mcr_on_cube(
                cube=hmap.cube,
                n_components=3,
                max_iter=500,
                random_state=0
            )
            
            if mcr_result and mcr_result.get("converged"):
                outputs["mcr"] = {
                    "n_components": 3,
                    "reconstruction_error": mcr_result["lack_of_fit"] * 100,
                    "n_iterations": mcr_result["n_iterations"],
                    "converged": True,
                    "lack_of_fit": mcr_result["lack_of_fit"],
                    "C_": mcr_result.get("concentrations"),
                    "ST_": mcr_result.get("component_spectra")
                }
                logger.info(f"MCR-ALS converged in {mcr_result['n_iterations']} iterations")
            else:
                logger.warning("MCR-ALS did not converge")
        except Exception as e:
            logger.warning(f"MCR-ALS failed: {e}")
    elif run_mcr:
        logger.warning("MCR-ALS skipped (pymcr not available)")
    
    # Generate plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save some basic plots for the report
    import matplotlib.pyplot as plt
    
    # PCA scree plot
    fig, ax = plt.subplots(figsize=(8, 6))
    explained = outputs["pca"]["explained_variance"]
    ax.plot(range(1, len(explained)+1), explained, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax.set_title('PCA Scree Plot', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "pca_scree.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # PCA loadings
    fig, axes = plt.subplots(len(explained), 1, figsize=(10, 3*len(explained)))
    if len(explained) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(hmap.energy, outputs["pca"]["components"][i], linewidth=2)
        ax.set_xlabel('Binding Energy (eV)', fontsize=11)
        ax.set_ylabel(f'PC{i+1} Loading', fontsize=11)
        ax.set_title(f'PC{i+1} ({explained[i]*100:.1f}% variance)', fontsize=12)
        ax.grid(alpha=0.3)
        ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(plots_dir / "pca_loadings.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # PCA score maps
    score_maps = outputs["pca"]["score_maps"]
    for i in range(score_maps.shape[2]):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(score_maps[:, :, i], cmap='viridis', aspect='auto')
        ax.set_title(f'PC{i+1} Score Map', fontsize=14)
        ax.set_xlabel('X Position', fontsize=11)
        ax.set_ylabel('Y Position', fontsize=11)
        plt.colorbar(im, ax=ax, label='Score')
        plt.tight_layout()
        plt.savefig(plots_dir / f"pca_score_map_{i}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Cluster map
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(outputs["cluster_labels"], cmap='tab10', aspect='auto')
    ax.set_title('Spatial Cluster Map', fontsize=14)
    ax.set_xlabel('X Position', fontsize=11)
    ax.set_ylabel('Y Position', fontsize=11)
    plt.colorbar(im, ax=ax, label='Cluster ID')
    plt.tight_layout()
    plt.savefig(plots_dir / "cluster_map.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Cluster spectra
    fig, ax = plt.subplots(figsize=(10, 6))
    for info in outputs["cluster_info"]:
        if info["mean_spec"] is not None:
            ax.plot(hmap.energy, info["mean_spec"], 
                   label=f'Cluster {info["cluster"]+1} (n={info["size"]})',
                   linewidth=2)
    ax.set_xlabel('Binding Energy (eV)', fontsize=12)
    ax.set_ylabel('Intensity (counts)', fontsize=12)
    ax.set_title('Mean Spectra by Cluster', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(plots_dir / "cluster_spectra.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # MSE map
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(outputs["mse_map"], cmap='hot', aspect='auto')
    ax.set_title('Mean Squared Error Map', fontsize=14)
    ax.set_xlabel('X Position', fontsize=11)
    ax.set_ylabel('Y Position', fontsize=11)
    plt.colorbar(im, ax=ax, label='MSE')
    plt.tight_layout()
    plt.savefig(plots_dir / "mse_map.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Shift map
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(outputs["shift_map"], cmap='RdBu_r', aspect='auto', 
                   vmin=-0.5, vmax=0.5)
    ax.set_title('Energy Shift Map', fontsize=14)
    ax.set_xlabel('X Position', fontsize=11)
    ax.set_ylabel('Y Position', fontsize=11)
    plt.colorbar(im, ax=ax, label='Shift (eV)')
    plt.tight_layout()
    plt.savefig(plots_dir / "shift_map.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("Plots saved successfully")
    
    # Prepare data for report generation using helper functions
    ny, nx, nE = hmap.shape
    hmap_metadata = {
        "region": hmap.metadata.region or "C1s",
        "file": hmap.metadata.source_file or "synthetic_map.txt",
        "format": hmap.metadata.source_format or "synthetic",
        "nx": nx,
        "ny": ny,
        "total_pixels": nx * ny,
        "n_energy_points": nE,
        "energy_min": float(hmap.energy.min()),
        "energy_max": float(hmap.energy.max())
    }
    
    # Use report_generator helper functions to format data
    pca_report_data = create_pca_report_data(outputs["pca"]) if "pca" in outputs else None
    mcr_report_data = create_mcr_report_data(outputs["mcr"]) if "mcr" in outputs else None
    
    # Format cluster results for reporting
    cluster_report_input = {
        "n_clusters": outputs.get("n_clusters", 0),
        "silhouette_score": outputs.get("silhouette_score", 0.0),
        "cluster_info": outputs.get("cluster_info", [])
    } if "cluster_labels" in outputs else None
    cluster_report_data = create_cluster_report_data(cluster_report_input) if cluster_report_input else None
    
    # Prepare pixel diagnostics for reporting
    pixel_diagnostics = {
        "mean_mse": float(np.mean(outputs["mse_map"])),
        "max_mse": float(np.max(outputs["mse_map"])),
        "problematic_pixels": int(np.sum(outputs["mse_map"] > 0.1))
    } if "mse_map" in outputs else None
    
    # Prepare plot paths
    plot_paths = {
        "pca_scree": str(plots_dir / "pca_scree.png"),
        "pca_loadings": str(plots_dir / "pca_loadings.png"),
        "cluster_map": str(plots_dir / "cluster_map.png"),
        "cluster_spectra": str(plots_dir / "cluster_spectra.png"),
        "mse_map": str(plots_dir / "mse_map.png"),
        "shift_map": str(plots_dir / "shift_map.png")
    }
    
    # Add PCA score maps
    for i in range(3):
        plot_paths[f"pca_score_map_{i}"] = str(plots_dir / f"pca_score_map_{i}.png")
    
    # Add MCR plots if available
    if "mcr" in outputs:
        # Generate MCR spectra plot
        fig, ax = plt.subplots(figsize=(10, 6))
        mcr_spectra = outputs["mcr"].get("ST_")
        if mcr_spectra is not None and len(mcr_spectra) > 0:
            for i in range(min(3, mcr_spectra.shape[0])):
                ax.plot(hmap.energy, mcr_spectra[i, :], 
                       label=f'Component {i+1}', linewidth=2)
            ax.set_xlabel('Binding Energy (eV)', fontsize=12)
            ax.set_ylabel('Intensity (a.u.)', fontsize=12)
            ax.set_title('MCR-ALS Resolved Pure Component Spectra', fontsize=14)
            ax.legend()
            ax.grid(alpha=0.3)
            ax.invert_xaxis()
            plt.tight_layout()
            plt.savefig(plots_dir / "mcr_spectra.png", dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths["mcr_spectra"] = str(plots_dir / "mcr_spectra.png")
            
            # Generate MCR concentration maps
            mcr_conc = outputs["mcr"].get("C_")
            if mcr_conc is not None and len(mcr_conc) > 0:
                conc_maps = mcr_conc.reshape(ny, nx, -1)
                for i in range(min(3, conc_maps.shape[2])):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(conc_maps[:, :, i], cmap='viridis', aspect='auto')
                    ax.set_title(f'MCR Component {i+1} Concentration', fontsize=14)
                    ax.set_xlabel('X Position', fontsize=11)
                    ax.set_ylabel('Y Position', fontsize=11)
                    plt.colorbar(im, ax=ax, label='Concentration')
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"mcr_concentration_{i}.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    plot_paths[f"mcr_concentration_{i}"] = str(plots_dir / f"mcr_concentration_{i}.png")
                
                # Generate combined RGB concentration map (if 3 components)
                if conc_maps.shape[2] >= 3:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    # Normalize each component to 0-1
                    rgb_map = np.zeros((ny, nx, 3))
                    for i in range(3):
                        comp = conc_maps[:, :, i]
                        comp_norm = (comp - comp.min()) / (comp.max() - comp.min() + 1e-10)
                        rgb_map[:, :, i] = comp_norm
                    ax.imshow(rgb_map, aspect='auto')
                    ax.set_title('Combined MCR Concentration Map (RGB)', fontsize=14)
                    ax.set_xlabel('X Position', fontsize=11)
                    ax.set_ylabel('Y Position', fontsize=11)
                    plt.tight_layout()
                    plt.savefig(plots_dir / "combined_concentration_map.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    plot_paths["combined_concentration_map"] = str(plots_dir / "combined_concentration_map.png")
    
    # Generate report
    logger.info(f"Generating {format.upper()} report...")
    report_path = generate_comprehensive_report(
        output_dir=output_dir,
        hmap_metadata=hmap_metadata,
        pca_results=pca_report_data,
        mcr_results=mcr_report_data,
        cluster_results=cluster_report_data,
        pixel_diagnostics=pixel_diagnostics,
        plot_paths=plot_paths,
        format=format
    )
    
    if report_path:
        logger.info(f"✓ Report generated successfully: {report_path}")
        logger.info(f"  File size: {report_path.stat().st_size / 1024:.1f} KB")
        return report_path
    else:
        logger.error("✗ Report generation failed")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test XPS map report generator")
    parser.add_argument("--file", type=str, default=None, 
                       help="Path to real hyperspectral map file (optional)")
    parser.add_argument("--format", type=str, default="html", choices=["html", "pdf"],
                       help="Report format (html or pdf)")
    parser.add_argument("--nx", type=int, default=20, help="Synthetic map X size")
    parser.add_argument("--ny", type=int, default=20, help="Synthetic map Y size")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: 05_map_data/test_report)")
    parser.add_argument("--no-mcr", action="store_true", help="Skip MCR-ALS")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(verbose=args.verbose)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = OUTPUT_DIR / "test_report"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create map
    if args.file:
        logger.info(f"Loading map from file: {args.file}")
        try:
            hmap = parse_map_with_config(args.file)
            if not isinstance(hmap, HyperspectralMap):
                logger.error("File is not a hyperspectral map")
                return
        except Exception as e:
            logger.error(f"Failed to load map: {e}")
            return
    else:
        logger.info("Creating synthetic hyperspectral map for testing...")
        hmap = create_synthetic_hyperspectral_map(nx=args.nx, ny=args.ny)
    
    # Run test
    report_path = test_report_generation(
        hmap=hmap,
        output_dir=output_dir,
        format=args.format,
        run_mcr=not args.no_mcr
    )
    
    if report_path:
        logger.info("\n" + "="*60)
        logger.info("SUCCESS! Report generated at:")
        logger.info(f"  {report_path.absolute()}")
        logger.info("="*60)
        
        if args.format == "html":
            logger.info("\nOpen in browser:")
            logger.info(f"  file:///{report_path.absolute()}")
    else:
        logger.error("\nReport generation failed. Check logs above for errors.")


if __name__ == "__main__":
    main()
