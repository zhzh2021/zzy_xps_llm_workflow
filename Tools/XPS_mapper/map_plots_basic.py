"""
Map visualization utilities for XPS mapper.

Common plotting functions for 2D and hyperspectral maps.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import logging

logger = logging.getLogger("xps_map")


def ensure_output_dir(path: Path):
    """Create output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig, outpath: Path, dpi: int = 200):
    """
    Save figure to file and close it.
    
    Args:
        fig: Matplotlib figure object
        outpath: Path to save the figure
        dpi: Resolution in dots per inch
    """
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_2d_overview(
    parsed,
    denoised: np.ndarray,
    mask: np.ndarray,
    net_map: Optional[np.ndarray],
    ratio_map: Optional[np.ndarray],
    threshold: float,
    output_dir: Path,
    show: bool = False
):
    """
    Create overview plots for 2D single-energy map analysis.
    
    Args:
        parsed: Map2D object with metadata
        denoised: Denoised intensity map
        mask: Binary segmentation mask
        net_map: Net intensity map (on-peak minus off-peak)
        ratio_map: Ratio map (net/off-peak)
        threshold: Threshold value used for segmentation
        output_dir: Directory to save plots
        show: Whether to display plots interactively
    """
    ensure_output_dir(output_dir)
    base_name = Path(parsed.metadata.source_file).stem if parsed.metadata.source_file else "map"
    region = parsed.metadata.region or "unknown"

    # Figure: raw vs denoised, mask overlay, histogram with threshold
    fig1, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    im0 = axs[0, 0].imshow(parsed.data, cmap='inferno')
    axs[0, 0].set_title("Raw map")
    fig1.colorbar(im0, ax=axs[0, 0])

    im1 = axs[0, 1].imshow(denoised, cmap='inferno')
    axs[0, 1].set_title("Denoised map")
    fig1.colorbar(im1, ax=axs[0, 1])

    im2 = axs[1, 0].imshow(denoised, cmap='gray')
    axs[1, 0].contour(mask.astype(float), levels=[0.5], colors='r', linewidths=0.8)
    axs[1, 0].set_title("Mask overlay")

    axs[1, 1].hist(denoised.ravel(), bins=50, color='steelblue')
    axs[1, 1].axvline(threshold, color='red', linestyle='--', label=f"thr={threshold:.2f}")
    axs[1, 1].set_title("Histogram with threshold")
    axs[1, 1].legend()

    save_fig(fig1, output_dir / f"{base_name}_{region}_2D_overview.png")

    # Optional: net/ratio maps
    if net_map is not None:
        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
        imn = ax2[0].imshow(net_map, cmap='inferno')
        ax2[0].set_title("Net map")
        fig2.colorbar(imn, ax=ax2[0])
        
        if ratio_map is not None:
            imr = ax2[1].imshow(ratio_map, cmap='viridis')
            ax2[1].set_title("Ratio map")
            fig2.colorbar(imr, ax=ax2[1])
            
        save_fig(fig2, output_dir / f"{base_name}_{region}_net_ratio.png")

    if show:
        plt.show()


def plot_area_maps(
    area_maps: dict,
    region: str,
    base_name: str,
    output_dir: Path,
    cmap: str = 'inferno'
):
    """
    Plot multiple area/intensity maps in a grid.
    
    Args:
        area_maps: Dictionary of {component_name: 2D array}
        region: Region name for title
        base_name: Base filename for output
        output_dir: Directory to save plot
        cmap: Colormap name
    """
    n_components = len(area_maps)
    cols = min(3, n_components)
    rows = int(np.ceil(n_components / cols))
    
    figA, axsA = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axsA = np.atleast_2d(axsA)
    
    for idx, (comp_name, area_map) in enumerate(area_maps.items()):
        r = idx // cols
        c = idx % cols
        im = axsA[r, c].imshow(area_map, cmap=cmap)
        axsA[r, c].set_title(f"{comp_name}")
        figA.colorbar(im, ax=axsA[r, c])
        
    # Hide unused axes
    for k in range(n_components, rows*cols):
        r = k // cols
        c = k % cols
        axsA[r, c].axis('off')
        
    save_fig(figA, output_dir / f"{base_name}_{region}_area_maps.png")


def plot_shift_mse_maps(
    shift_map: np.ndarray,
    mse_map: np.ndarray,
    region: str,
    base_name: str,
    output_dir: Path
):
    """
    Plot energy shift and fitting MSE maps side by side.
    
    Args:
        shift_map: 2D array of energy shifts
        mse_map: 2D array of fit mean squared errors
        region: Region name for title
        base_name: Base filename for output
        output_dir: Directory to save plot
    """
    figSM, axsSM = plt.subplots(1, 2, figsize=(10, 4))
    
    ims = axsSM[0].imshow(shift_map, cmap='coolwarm')
    axsSM[0].set_title("Energy shift (eV)")
    figSM.colorbar(ims, ax=axsSM[0])
    
    imm = axsSM[1].imshow(mse_map, cmap='magma')
    axsSM[1].set_title("Fit MSE")
    figSM.colorbar(imm, ax=axsSM[1])
    
    save_fig(figSM, output_dir / f"{base_name}_{region}_shift_mse.png")


def plot_average_spectrum(
    energy: np.ndarray,
    avg_spectrum: np.ndarray,
    baseline: Optional[np.ndarray],
    corrected: Optional[np.ndarray],
    region: str,
    base_name: str,
    output_dir: Path
):
    """
    Plot average spectrum with optional baseline and correction.
    
    Args:
        energy: Energy axis
        avg_spectrum: Average spectrum across all pixels
        baseline: Baseline array (optional)
        corrected: Baseline-corrected spectrum (optional)
        region: Region name for title
        base_name: Base filename for output
        output_dir: Directory to save plot
    """
    figS, axS = plt.subplots(1, 1, figsize=(8, 4))
    
    axS.plot(energy, avg_spectrum, label="avg")
    
    if baseline is not None:
        axS.plot(energy, baseline, label="baseline", linestyle="--")
        
    if corrected is not None:
        axS.plot(energy, corrected, label="corrected")
        
    axS.set_xlabel("Energy (eV)")
    axS.set_ylabel("Intensity (a.u.)")
    axS.set_title(f"{region} average spectrum")
    axS.legend()
    axS.invert_xaxis()  # Reverse x-axis for binding energy convention
    
    save_fig(figS, output_dir / f"{base_name}_{region}_avg_spectrum.png")
