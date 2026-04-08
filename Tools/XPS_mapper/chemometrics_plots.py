"""
Chemometrics visualization utilities for XPS hyperspectral maps.

Functions for PRE images, waterfall plots, and other chemometrics visualizations.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional
import logging

logger = logging.getLogger("xps_map")


def ensure_output_dir(path: Path):
    """Create output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig, outpath: Path, dpi: int = 200):
    """Save figure to file and close it."""
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_spectra_waterfall(
    cube: np.ndarray,
    energy: np.ndarray,
    output_dir: Path,
    base_name: str = "map",
    region: str = "",
    n_spectra: int = 20,
    plot_mode: str = "overlay",
    show: bool = False
):
    """
    Create waterfall or overlay plot of spectra from hyperspectral map.
    
    Useful for initial data inspection:
    - If spectra are identical → no spatial variation → skip chemometrics
    - If spectra vary → proceed with PCA/clustering analysis
    
    Args:
        cube: (ny, nx, nE) hyperspectral data
        energy: (nE,) energy axis
        output_dir: Directory to save plot
        base_name: Base filename
        region: Region name for title
        n_spectra: Number of spectra to plot (uniformly sampled)
        plot_mode: 'overlay' or 'waterfall'
        show: Display plot interactively
    """
    ensure_output_dir(output_dir)
    ny, nx, nE = cube.shape
    total_pixels = ny * nx
    
    # Uniformly sample pixels across the map
    if n_spectra > total_pixels:
        n_spectra = total_pixels
    
    # Sample indices uniformly
    sample_indices = np.linspace(0, total_pixels - 1, n_spectra, dtype=int)
    
    # Extract spectra
    X = cube.reshape(total_pixels, nE)
    sampled_spectra = X[sample_indices, :]
    
    # Compute variability metric
    std_per_energy = np.std(sampled_spectra, axis=0)
    mean_std = np.mean(std_per_energy)
    max_std = np.max(std_per_energy)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plot_mode == "waterfall":
        # Waterfall plot with vertical offset
        max_intensity = np.max(sampled_spectra)
        offset = max_intensity * 0.3  # 30% offset between spectra
        
        for i, spec in enumerate(sampled_spectra):
            ax.plot(energy, spec + i * offset, alpha=0.7, linewidth=0.8,
                   color=cm.viridis(i / n_spectra))
        
        ax.set_ylabel('Intensity (offset)', fontsize=16)
        ax.set_title(f'{region} Spectra Waterfall (n={n_spectra})\n'
                    f'Variability: mean σ={mean_std:.1f}, max σ={max_std:.1f}',
                    fontsize=15)
    else:
        # Overlay plot
        for i, spec in enumerate(sampled_spectra):
            ax.plot(energy, spec, alpha=0.5, linewidth=0.8,
                   color=cm.viridis(i / n_spectra))
        
        ax.set_ylabel('Intensity (a.u.)', fontsize=16)
        ax.set_title(f'{region} Spectra Overlay (n={n_spectra})\n'
                    f'Variability: mean σ={mean_std:.1f}, max σ={max_std:.1f}',
                    fontsize=15)

    ax.set_xlabel('Binding Energy (eV)', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Reverse x-axis for binding energy convention
    
    # Add interpretation guidance as text
    if mean_std < 10:
        guidance = "LOW variability → Consider skipping chemometrics"
        color = 'orange'
    else:
        guidance = "SUFFICIENT variability → Proceed with analysis"
        color = 'green'
    
    ax.text(0.02, 0.98, guidance, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    # Save
    suffix = "waterfall" if plot_mode == "waterfall" else "overlay"
    outpath = output_dir / f"{base_name}_{region}_spectra_{suffix}.png"
    save_fig(fig, outpath)
    logger.info(f"Saved spectra {suffix} plot: {outpath.name}")
    
    if show:
        plt.show()
    
    return {"mean_std": mean_std, "max_std": max_std}


def plot_pre_image(
    pre_image: np.ndarray,
    output_dir: Path,
    base_name: str = "map",
    region: str = "",
    x_axis: Optional[np.ndarray] = None,
    y_axis: Optional[np.ndarray] = None,
    show: bool = False
):
    """
    Plot Pattern Recognition Entropy (PRE) as a 2D image.
    
    PRE shows spectral complexity/diversity at each pixel:
    - High PRE (hot colors) = complex/mixed spectrum
    - Low PRE (cool colors) = pure/simple spectrum
    
    Args:
        pre_image: (ny, nx) PRE values
        output_dir: Directory to save plot
        base_name: Base filename
        region: Region name for title
        x_axis: Optional x-coordinates for extent
        y_axis: Optional y-coordinates for extent
        show: Display plot interactively
    """
    ensure_output_dir(output_dir)
    
    ny, nx = pre_image.shape
    
    # Statistics
    mean_pre = np.mean(pre_image)
    std_pre = np.std(pre_image)
    min_pre = np.min(pre_image)
    max_pre = np.max(pre_image)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: PRE image
    ax = axes[0]
    extent = None
    if x_axis is not None and y_axis is not None:
        extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]
    
    im = ax.imshow(pre_image, cmap='jet', interpolation='nearest',
                   aspect='auto', origin='lower', extent=extent)
    ax.set_title(f'{region} Pattern Recognition Entropy\n'
                f'Mean: {mean_pre:.2f} ± {std_pre:.2f}',
                fontsize=11)
    
    if extent:
        ax.set_xlabel('X (μm)', fontsize=15)
        ax.set_ylabel('Y (μm)', fontsize=15)
    else:
        ax.set_xlabel('X pixel', fontsize=15)
        ax.set_ylabel('Y pixel', fontsize=15)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PRE (Shannon Entropy)', fontsize=15)

    # Right panel: Histogram
    ax = axes[1]
    ax.hist(pre_image.ravel(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(mean_pre, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_pre:.2f}')
    ax.axvline(mean_pre - std_pre, color='orange', linestyle=':', linewidth=1.5, label=f'±1σ')
    ax.axvline(mean_pre + std_pre, color='orange', linestyle=':', linewidth=1.5)

    ax.set_xlabel('PRE Value', fontsize=15)
    ax.set_ylabel('Frequency', fontsize=15)
    ax.set_title(f'PRE Distribution\nRange: [{min_pre:.2f}, {max_pre:.2f}]', fontsize=15)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.3)
    
    # Add interpretation guidance
    # PRE typically ranges 0-5 for XPS data
    if std_pre < 0.3:
        guidance = "LOW heterogeneity → Uniform sample"
    elif std_pre < 0.8:
        guidance = "MODERATE heterogeneity"
    else:
        guidance = "HIGH heterogeneity → Complex phases"
    
    ax.text(0.02, 0.98, guidance, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save
    outpath = output_dir / f"{base_name}_{region}_PRE_map.png"
    save_fig(fig, outpath)
    logger.info(f"Saved PRE map: {outpath.name}")
    
    if show:
        plt.show()
    
    return {
        "mean": mean_pre,
        "std": std_pre,
        "min": min_pre,
        "max": max_pre
    }


def plot_mcr_components(
    energy: np.ndarray,
    component_spectra: np.ndarray,
    conc_maps: np.ndarray,
    method: str,
    output_dir: Path,
    base_name: str = "map",
    region: str = "",
    x_axis: Optional[np.ndarray] = None,
    y_axis: Optional[np.ndarray] = None,
    show: bool = False
):
    """
    Plot MCR-ALS or NMF component spectra and concentration maps.
    
    Args:
        energy: (nE,) energy axis
        component_spectra: (nE, n_components) pure component spectra
        conc_maps: (ny, nx, n_components) concentration maps
        method: 'MCR-ALS' or 'NMF'
        output_dir: Directory to save plots
        base_name: Base filename
        region: Region name
        x_axis: Optional x-coordinates
        y_axis: Optional y-coordinates
        show: Display plot interactively
    """
    ensure_output_dir(output_dir)
    
    n_components = component_spectra.shape[1]
    ny, nx = conc_maps.shape[:2]
    
    extent = None
    if x_axis is not None and y_axis is not None:
        extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]
    
    # Create figure with component spectra and maps
    fig = plt.figure(figsize=(14, 4 * n_components))
    
    for i in range(n_components):
        # Spectrum plot
        ax_spec = plt.subplot(n_components, 2, 2*i + 1)
        ax_spec.plot(energy, component_spectra[:, i], linewidth=2, color='C0')
        ax_spec.set_xlabel('Binding Energy (eV)', fontsize=15)
        ax_spec.set_ylabel('Intensity (a.u.)', fontsize=15)
        ax_spec.set_title(f'{method} Component {i+1} Spectrum', fontsize=15)
        ax_spec.grid(True, alpha=0.3)
        ax_spec.invert_xaxis()
        
        # Concentration map
        ax_map = plt.subplot(n_components, 2, 2*i + 2)
        im = ax_map.imshow(conc_maps[:, :, i], cmap='viridis',
                          interpolation='nearest', aspect='auto',
                          origin='lower', extent=extent)
        ax_map.set_title(f'{method} Component {i+1} Concentration', fontsize=11)
        
        if extent:
            ax_map.set_xlabel('X (μm)', fontsize=15)
            ax_map.set_ylabel('Y (μm)', fontsize=15)
        else:
            ax_map.set_xlabel('X pixel', fontsize=15)
            ax_map.set_ylabel('Y pixel', fontsize=15)

        cbar = plt.colorbar(im, ax=ax_map)
        cbar.set_label('Concentration', fontsize=15)

    plt.suptitle(f'{region} {method} Analysis (n={n_components})', fontsize=15, y=0.995)

    # Save
    outpath = output_dir / f"{base_name}_{region}_MCR_components.png"
    save_fig(fig, outpath)
    logger.info(f"Saved MCR components: {outpath.name}")
    
    if show:
        plt.show()


def plot_scree(
    explained_variance: np.ndarray,
    n_selected: int,
    variance_threshold: float,
    output_dir: Path,
    base_name: str = "map",
    region: str = "",
    show: bool = False
):
    """
    Create scree plot showing cumulative explained variance from PCA.
    
    Visualizes component selection rationale for MCR analysis.
    
    Args:
        explained_variance: (k,) explained variance ratios from PCA
        n_selected: Number of components selected
        variance_threshold: Threshold used (e.g., 0.99)
        output_dir: Directory to save plot
        base_name: Base filename
        region: Region name for title
        show: Display plot interactively
    """
    ensure_output_dir(output_dir)
    
    n_components = len(explained_variance)
    cumulative_variance = np.cumsum(explained_variance)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Individual variance plot
    ax1.bar(range(1, n_components + 1), explained_variance * 100,
           alpha=0.7, color='steelblue', label='Individual')
    ax1.axvline(n_selected, color='red', linestyle='--', linewidth=2,
               label=f'Selected: {n_selected}')
    ax1.set_xlabel('Principal Component', fontsize=11)
    ax1.set_ylabel('Explained Variance (%)', fontsize=11)
    ax1.set_title('Individual Variance per Component', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Cumulative variance plot
    ax2.plot(range(1, n_components + 1), cumulative_variance * 100,
            marker='o', linewidth=2, markersize=6, color='steelblue')
    ax2.axhline(variance_threshold * 100, color='green', linestyle='--',
               linewidth=2, label=f'Threshold: {variance_threshold*100:.0f}%')
    ax2.axvline(n_selected, color='red', linestyle='--', linewidth=2,
               label=f'Selected: {n_selected}')
    ax2.set_xlabel('Number of Components', fontsize=15)
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=15)
    ax2.set_title(f'Cumulative Variance (Selected: {cumulative_variance[n_selected-1]*100:.1f}%)',
                 fontsize=15)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.suptitle(f'{region} PCA Scree Analysis', fontsize=15, y=1.00)
    
    # Save
    outpath = output_dir / f"{base_name}_{region}_scree_plot.png"
    save_fig(fig, outpath)
    logger.info(f"Saved scree plot: {outpath.name}")
    
    if show:
        plt.show()


def plot_mcr_quality_metrics(
    quality_metrics: dict,
    output_dir: Path,
    base_name: str = "map",
    region: str = "",
    x_axis: Optional[np.ndarray] = None,
    y_axis: Optional[np.ndarray] = None,
    show: bool = False
):
    """
    Visualize MCR quality metrics: R², lack-of-fit (LOF), residual maps.
    
    Helps validate MCR decomposition:
    - High R² (>0.9) → Good fit
    - Low LOF (<0.1) → Low residuals
    - Spatial patterns in residuals → May need more components
    
    Args:
        quality_metrics: Dict from compute_mcr_quality_metrics() with:
            - r2_map: (ny, nx) R² per pixel
            - lof_map: (ny, nx) lack-of-fit per pixel
            - total_r2: float, overall R²
            - mean_lof: float, average LOF
        output_dir: Directory to save plot
        base_name: Base filename
        region: Region name for title
        x_axis: X-axis coordinates (optional)
        y_axis: Y-axis coordinates (optional)
        show: Display plot interactively
    """
    ensure_output_dir(output_dir)
    
    r2_map = quality_metrics["r2_map"]
    lof_map = quality_metrics["lof_map"]
    total_r2 = quality_metrics["total_r2"]
    mean_lof = quality_metrics["mean_lof"]
    
    ny, nx = r2_map.shape
    
    # Setup spatial extent
    if x_axis is not None and y_axis is not None:
        extent = [x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]]
        xlabel, ylabel = "X (μm)", "Y (μm)"
    else:
        extent = [0, nx, ny, 0]
        xlabel, ylabel = "X (pixels)", "Y (pixels)"
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² map
    im1 = axes[0].imshow(r2_map, cmap='RdYlGn', vmin=0, vmax=1, 
                        extent=extent, aspect='auto', origin='upper')
    axes[0].set_title(f'R² Map (Overall: {total_r2:.3f})', fontsize=12, weight='bold')
    axes[0].set_xlabel(xlabel, fontsize=15)
    axes[0].set_ylabel(ylabel, fontsize=15)
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('R²', fontsize=15)

    # Lack-of-fit map
    im2 = axes[1].imshow(lof_map, cmap='hot_r', vmin=0, vmax=np.percentile(lof_map, 98),
                        extent=extent, aspect='auto', origin='upper')
    axes[1].set_title(f'Lack-of-Fit Map (Mean: {mean_lof:.4f})', fontsize=12, weight='bold')
    axes[1].set_xlabel(xlabel, fontsize=15)
    axes[1].set_ylabel(ylabel, fontsize=15)
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('LOF', fontsize=15)

    plt.suptitle(f'{region} MCR Quality Metrics', fontsize=15, y=0.98)
    
    # Save
    outpath = output_dir / f"{base_name}_{region}_MCR_quality.png"
    save_fig(fig, outpath)
    logger.info(f"Saved MCR quality plot: {outpath.name} (R²={total_r2:.3f}, LOF={mean_lof:.4f})")
    
    if show:
        plt.show()


def plot_cluster_validation(
    cluster_results: dict,
    energy: np.ndarray,
    valid_mask: np.ndarray,
    output_dir: Path,
    base_name: str = "map",
    region: str = "",
    show: bool = False
):
    """
    Visualize cluster validation results showing valid vs invalid clusters.
    
    Args:
        cluster_results: Dict with 'labels', 'cluster_info'
        energy: (p,) energy axis
        valid_mask: (m*n,) boolean array from validate_cluster_spectra()
        output_dir: Directory to save plot
        base_name: Base filename
        region: Region name for title
        show: Display plot interactively
    """
    ensure_output_dir(output_dir)
    
    labels = cluster_results["labels"]
    cluster_info = cluster_results["cluster_info"]
    n_clusters = len(cluster_info)
    
    # Flatten labels if needed to match valid_mask shape
    labels_flat = labels.ravel() if labels.ndim > 1 else labels
    
    # Determine which clusters are valid
    valid_clusters = []
    for i in range(n_clusters):
        cluster_pixels = (labels_flat == i)
        if np.any(valid_mask & cluster_pixels):
            valid_clusters.append(i)
    
    invalid_clusters = [i for i in range(n_clusters) if i not in valid_clusters]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot valid clusters (solid lines)
    for i in valid_clusters:
        mean_spec = cluster_info[i]["mean_spec"]
        if mean_spec is not None:
            ax.plot(energy, mean_spec, linewidth=2, label=f'Cluster {i} ✓', alpha=0.8)
    
    # Plot invalid clusters (dashed lines)
    for i in invalid_clusters:
        mean_spec = cluster_info[i]["mean_spec"]
        if mean_spec is not None:
            ax.plot(energy, mean_spec, linewidth=2, linestyle='--', 
                   label=f'Cluster {i} ✗ (invalid)', alpha=0.5, color='red')
    
    ax.set_xlabel('Binding Energy (eV)', fontsize=16)
    ax.set_ylabel('Intensity (counts)', fontsize=16)
    ax.set_title(f'{region} Cluster Validation: {len(valid_clusters)}/{n_clusters} Valid', 
                fontsize=16, weight='bold')
    ax.legend(loc='best', fontsize=15)
    ax.grid(alpha=0.3)
    
    # Invert x-axis for binding energy convention
    ax.invert_xaxis()
    
    # Save
    outpath = output_dir / f"{base_name}_{region}_cluster_validation.png"
    save_fig(fig, outpath)
    logger.info(f"Saved cluster validation plot: {outpath.name}")
    
    if show:
        plt.show()
