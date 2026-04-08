"""
PCA and NMF component visualization for hyperspectral maps.

Functions for plotting dimensionality reduction results.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import logging

logger = logging.getLogger("xps_map")


def save_fig(fig, outpath: Path, dpi: int = 200):
    """Save figure to file and close it."""
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_pca_components(
    energy: np.ndarray,
    pca_info: Dict,
    region: str,
    base_name: str,
    output_dir: Path,
    show: bool = False
):
    """
    Plot PCA component spectra and score maps.
    
    Args:
        energy: Energy axis array
        pca_info: Dictionary with 'components', 'score_maps', 'explained_variance'
        region: Region name for title
        base_name: Base filename for output
        output_dir: Directory to save plot
        show: Whether to display plots interactively
    """
    comps = pca_info['components']
    scores = pca_info['score_maps']
    var = pca_info['explained_variance']
    n_pca = comps.shape[0]
    
    figPC, axsPC = plt.subplots(n_pca, 2, figsize=(10, 3*n_pca))
    axsPC = np.atleast_2d(axsPC)
    
    for i in range(n_pca):
        # Plot component spectrum (loading)
        axsPC[i, 0].plot(energy, comps[i], color='black')
        axsPC[i, 0].set_title(f"PC{i+1} ({var[i]*100:.1f}% var)")
        axsPC[i, 0].set_xlabel("Energy (eV)")
        axsPC[i, 0].set_ylabel("Loading")
        axsPC[i, 0].invert_xaxis()  # Reverse x-axis for binding energy convention
        
        # Plot score map (spatial distribution)
        im = axsPC[i, 1].imshow(scores[:, :, i], cmap='plasma')
        axsPC[i, 1].set_title(f"PC{i+1} score map")
        figPC.colorbar(im, ax=axsPC[i, 1])
        
    save_fig(figPC, output_dir / f"{base_name}_{region}_pca.png")
    
    if show:
        plt.show()


def plot_pca_scores(
    score_maps: np.ndarray,
    explained_variance: np.ndarray,
    output_dir: Path,
    base_name: str = "pca_scores",
    cmap: str = 'RdBu_r'
):
    """
    Plot PCA score maps in a grid.
    
    Args:
        score_maps: 3D array (ny, nx, n_components)
        explained_variance: Array of explained variance ratios
        output_dir: Directory to save plot
        base_name: Base filename for output
        cmap: Colormap name
    """
    n_components = score_maps.shape[2]
    cols = min(3, n_components)
    rows = int(np.ceil(n_components / cols))
    
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axs = np.atleast_2d(axs)
    
    for i in range(n_components):
        r = i // cols
        c = i % cols
        im = axs[r, c].imshow(score_maps[:, :, i], cmap=cmap)
        axs[r, c].set_title(f'PC{i+1} ({explained_variance[i]*100:.1f}%)')
        plt.colorbar(im, ax=axs[r, c])
        axs[r, c].set_xticks([])
        axs[r, c].set_yticks([])
        
    # Hide unused axes
    for k in range(n_components, rows*cols):
        r = k // cols
        c = k % cols
        axs[r, c].axis('off')
        
    save_fig(fig, output_dir / f"{base_name}.png")


def plot_pca_loadings(
    energy: np.ndarray,
    components: np.ndarray,
    explained_variance: np.ndarray,
    output_dir: Path,
    base_name: str = "pca_loadings"
):
    """
    Plot PCA loading spectra (components).
    
    Args:
        energy: Energy axis
        components: 2D array (n_components, n_energy_points)
        explained_variance: Array of explained variance ratios
        output_dir: Directory to save plot
        base_name: Base filename for output
    """
    n_components = components.shape[0]
    
    fig, axs = plt.subplots(n_components, 1, figsize=(8, 3*n_components))
    if n_components == 1:
        axs = [axs]
        
    for i in range(n_components):
        axs[i].plot(energy, components[i], 'b-')
        axs[i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axs[i].set_title(f'PC{i+1} Loading ({explained_variance[i]*100:.1f}% variance)')
        axs[i].set_xlabel('Binding Energy (eV)')
        axs[i].set_ylabel('Loading')
        
    save_fig(fig, output_dir / f"{base_name}.png")


def plot_nmf_components(
    energy: np.ndarray,
    nmf_info: Dict,
    region: str,
    base_name: str,
    output_dir: Path,
    show: bool = False
):
    """
    Plot NMF component spectra and abundance maps.
    
    Args:
        energy: Energy axis array
        nmf_info: Dictionary with 'components', 'abundance_maps'
        region: Region name for title
        base_name: Base filename for output
        output_dir: Directory to save plot
        show: Whether to display plots interactively
    """
    comps = nmf_info['components']
    abund = nmf_info['abundance_maps']
    n_nmf = comps.shape[0]
    
    figNM, axsNM = plt.subplots(n_nmf, 2, figsize=(10, 3*n_nmf))
    axsNM = np.atleast_2d(axsNM)
    
    for i in range(n_nmf):
        # Plot component spectrum
        axsNM[i, 0].plot(energy, comps[i], color='black')
        axsNM[i, 0].set_title(f"NMF comp {i+1}")
        axsNM[i, 0].set_xlabel("Energy (eV)")
        axsNM[i, 0].set_ylabel("Intensity")
        axsNM[i, 0].invert_xaxis()  # Reverse x-axis for binding energy convention
        
        # Plot abundance map (spatial distribution)
        im = axsNM[i, 1].imshow(abund[:, :, i], cmap='viridis')
        axsNM[i, 1].set_title(f"NMF abundance {i+1}")
        figNM.colorbar(im, ax=axsNM[i, 1])
        
    save_fig(figNM, output_dir / f"{base_name}_{region}_nmf.png")
    
    if show:
        plt.show()


def plot_scree(
    explained_variance: np.ndarray,
    output_dir: Path,
    base_name: str = "pca_scree"
):
    """
    Plot PCA scree plot showing explained variance.
    
    Args:
        explained_variance: Array of explained variance ratios
        output_dir: Directory to save plot
        base_name: Base filename for output
    """
    n_components = len(explained_variance)
    cumulative_var = np.cumsum(explained_variance)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Individual variance
    ax1.bar(range(1, n_components+1), explained_variance * 100)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance (%)')
    ax1.set_title('Explained Variance by Component')
    ax1.set_xticks(range(1, n_components+1))
    
    # Cumulative variance
    ax2.plot(range(1, n_components+1), cumulative_var * 100, 'bo-')
    ax2.axhline(y=80, color='r', linestyle='--', label='80% threshold')
    ax2.axhline(y=90, color='orange', linestyle='--', label='90% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance (%)')
    ax2.set_title('Cumulative Explained Variance')
    ax2.legend()
    ax2.set_xticks(range(1, n_components+1))
    ax2.grid(True, alpha=0.3)
    
    save_fig(fig, output_dir / f"{base_name}.png")
