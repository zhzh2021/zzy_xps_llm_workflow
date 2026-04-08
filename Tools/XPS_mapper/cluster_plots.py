"""
Cluster analysis visualization for hyperspectral maps.

Functions for plotting clustering results and dendrograms.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import logging

logger = logging.getLogger("xps_map")


def save_fig(fig, outpath: Path, dpi: int = 300):
    """Save figure to file and close it."""
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_cluster_analysis(
    map_data,
    cluster_results: Dict,
    output_dir: Path,
    show_plots: bool = False,
    base_name: str = "cluster_analysis"
):
    """
    Plot comprehensive cluster analysis results.
    
    Args:
        map_data: HyperspectralMap object with energy axis and shape
        cluster_results: Dictionary with 'pca', 'labels', 'cluster_info'
        output_dir: Directory to save plots
        show_plots: Whether to display plots interactively
        base_name: Base name for output files (for dataset-specific naming)
    """
    ny, nx = map_data.shape[:2]
    n_pca = cluster_results['pca']['components'].shape[0]
    n_clusters = max(info['cluster'] for info in cluster_results['cluster_info']) + 1
    
    # 1. PCA components and score maps
    fig_pca, axs_pca = plt.subplots(n_pca, 2, figsize=(12, 4*n_pca))
    if n_pca == 1:
        axs_pca = axs_pca.reshape(1, -1)
        
    # Get physical coordinates if available for score maps
    if hasattr(map_data, 'x_axis') and hasattr(map_data, 'y_axis'):
        extent_pca = [map_data.x_axis[0], map_data.x_axis[-1], 
                      map_data.y_axis[0], map_data.y_axis[-1]]
        xlabel_pca = 'X Position (μm)'
        ylabel_pca = 'Y Position (μm)'
    else:
        extent_pca = None
        xlabel_pca = 'X Pixel'
        ylabel_pca = 'Y Pixel'
    
    for i in range(n_pca):
        # Plot component spectrum
        axs_pca[i,0].plot(map_data.energy, cluster_results['pca']['components'][i])
        axs_pca[i,0].set_title(f'PC{i+1} spectrum')
        axs_pca[i,0].set_xlabel('Binding Energy (eV)')
        axs_pca[i,0].set_ylabel('Loading')
        axs_pca[i,0].invert_xaxis()  # Reverse x-axis for binding energy convention
        
        # Plot score map with physical coordinates
        im = axs_pca[i,1].imshow(cluster_results['pca']['score_maps'][:,:,i], 
                                 cmap='RdBu_r', origin='lower', aspect='auto',
                                 extent=extent_pca)
        plt.colorbar(im, ax=axs_pca[i,1])
        axs_pca[i,1].set_title(f'PC{i+1} scores')
        axs_pca[i,1].set_xlabel(xlabel_pca)
        axs_pca[i,1].set_ylabel(ylabel_pca)
    
    save_fig(fig_pca, output_dir / f'{base_name}_pca_analysis.png')
    if show_plots:
        plt.show()
    
    # 2. Cluster results
    fig_clust, (ax_map, ax_specs) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Cluster map - spatial distribution of chemical states with physical coordinates
    # Get physical coordinates if available
    if hasattr(map_data, 'x_axis') and hasattr(map_data, 'y_axis'):
        x_axis = map_data.x_axis
        y_axis = map_data.y_axis
        extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]
        xlabel = 'X Position (μm)'
        ylabel = 'Y Position (μm)'
    else:
        extent = None
        xlabel = 'X Pixel'
        ylabel = 'Y Pixel'
    
    im = ax_map.imshow(cluster_results['labels'], cmap='tab10', 
                       interpolation='nearest', origin='lower', aspect='auto',
                       extent=extent)
    plt.colorbar(im, ax=ax_map, label='Chemical State (Cluster ID)')
    ax_map.set_title('Spatial Distribution of Chemical States', fontsize=13, fontweight='bold')
    ax_map.set_xlabel(xlabel, fontsize=11)
    ax_map.set_ylabel(ylabel, fontsize=11)
    ax_map.grid(True, alpha=0.2, linestyle='--')
    
    # Mean spectra per cluster - representative chemical states
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for info in cluster_results['cluster_info']:
        if info['size'] > 0 and info['mean_spec'] is not None:
            label = f"Cluster {info['cluster']} (n={info['size']})"
            color = colors[info['cluster']]
            ax_specs.plot(map_data.energy, info['mean_spec'], label=label, color=color, linewidth=2)
    
    ax_specs.set_xlabel('Binding Energy (eV)', fontsize=11)
    ax_specs.set_ylabel('Intensity', fontsize=11)
    ax_specs.set_title('Representative Spectra of Chemical States', fontsize=13, fontweight='bold')
    ax_specs.legend(fontsize=9)
    ax_specs.grid(True, alpha=0.3)
    ax_specs.invert_xaxis()  # Reverse x-axis for binding energy convention
    
    save_fig(fig_clust, output_dir / f'{base_name}_cluster_analysis.png')
    if show_plots:
        plt.show()


def plot_cluster_map(
    labels: np.ndarray,
    output_dir: Path,
    base_name: str = "cluster_map",
    cmap: str = 'tab10',
    x_axis: Optional[np.ndarray] = None,
    y_axis: Optional[np.ndarray] = None,
    scan_area: Optional[tuple] = None
):
    """
    Plot cluster label map representing physical spatial distribution of chemical states.
    
    Args:
        labels: 2D array of cluster labels (ny, nx)
        output_dir: Directory to save plot
        base_name: Base filename for output
        cmap: Colormap name
        x_axis: Optional physical x-coordinates (μm)
        y_axis: Optional physical y-coordinates (μm)
        scan_area: Optional (width, height) in μm, e.g., (1000, 1000)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up extent for physical coordinates
    ny, nx = labels.shape
    if x_axis is not None and y_axis is not None:
        extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]
    elif scan_area is not None:
        # Use scan area dimensions
        extent = [0, scan_area[0], 0, scan_area[1]]
    else:
        # Default pixel coordinates
        extent = [0, nx, 0, ny]
    
    im = ax.imshow(labels, cmap=cmap, interpolation='nearest', 
                   origin='lower', extent=extent, aspect='auto')
    
    # Set title and labels based on coordinate system
    if x_axis is not None or scan_area is not None:
        ax.set_title('Spatial Distribution of Chemical States\n(Cluster Map)', 
                    fontsize=15, fontweight='bold')
        ax.set_xlabel('X Position (μm)', fontsize=15)
        ax.set_ylabel('Y Position (μm)', fontsize=15)
    else:
        ax.set_title('Spatial Distribution of Chemical Domains\n(Cluster Map)', 
                    fontsize=15, fontweight='bold')
        ax.set_xlabel('X Pixel', fontsize=15)
        ax.set_ylabel('Y Pixel', fontsize=15)
    
    # Add colorbar with discrete cluster IDs
    n_clusters = len(np.unique(labels))
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_clusters))
    cbar.set_label('Chemical Domain\n(Cluster ID)', fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=15)

    # Add grid for better spatial reference
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=15)
    
    save_fig(fig, output_dir / f"{base_name}.png")
    

def plot_cluster_spectra(
    energy: np.ndarray,
    cluster_info: list,
    output_dir: Path,
    base_name: str = "cluster_spectra",
    plot_type: str = "overlay"
):
    """
    Plot representative spectra for each cluster.
    
    Args:
        energy: Energy axis
        cluster_info: List of dicts with 'cluster', 'size', 'mean_spec'
        output_dir: Directory to save plot
        base_name: Base filename for output
        plot_type: 'overlay' or 'grid'
    """
    n_clusters = len(cluster_info)
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    if plot_type == "overlay":
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for info in cluster_info:
            if info['size'] > 0 and info['mean_spec'] is not None:
                label = f"Cluster {info['cluster']} (n={info['size']})"
                color = colors[info['cluster']]
                ax.plot(energy, info['mean_spec'], label=label, color=color, linewidth=2)
        
        ax.set_xlabel('Binding Energy (eV)',fontsize=15)
        ax.set_ylabel('Intensity',fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_title('Cluster Representative Spectra',fontsize=16)
        ax.legend(fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # Reverse x-axis for binding energy convention
        
    else:  # grid
        cols = min(3, n_clusters)
        rows = int(np.ceil(n_clusters / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 3*rows))
        axs = np.atleast_2d(axs)
        
        for idx, info in enumerate(cluster_info):
            r = idx // cols
            c = idx % cols
            if info['size'] > 0 and info['mean_spec'] is not None:
                color = colors[info['cluster']]
                axs[r, c].plot(energy, info['mean_spec'], color=color, linewidth=2)
                axs[r, c].set_title(f"Cluster {info['cluster']} (n={info['size']})")
                axs[r, c].set_xlabel('Binding Energy (eV)')
                axs[r, c].set_ylabel('Intensity')
                axs[r, c].grid(True, alpha=0.3)
                axs[r, c].invert_xaxis()  # Reverse x-axis for binding energy convention
            else:
                axs[r, c].axis('off')
                
        # Hide unused axes
        for k in range(n_clusters, rows*cols):
            r = k // cols
            c = k % cols
            axs[r, c].axis('off')
    
    save_fig(fig, output_dir / f"{base_name}.png")


def plot_cluster_scatter(
    score_maps: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    base_name: str = "cluster_scatter",
    pc_x: int = 0,
    pc_y: int = 1
):
    """
    Plot scatter of PCA scores colored by cluster.
    
    Args:
        score_maps: 3D array (ny, nx, n_components)
        labels: 2D array of cluster labels
        output_dir: Directory to save plot
        base_name: Base filename for output
        pc_x: PC index for x-axis
        pc_y: PC index for y-axis
    """
    ny, nx, n_pca = score_maps.shape
    
    # Flatten arrays
    scores_x = score_maps[:, :, pc_x].ravel()
    scores_y = score_maps[:, :, pc_y].ravel()
    labels_flat = labels.ravel()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n_clusters = len(np.unique(labels_flat))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        mask = labels_flat == cluster_id
        ax.scatter(scores_x[mask], scores_y[mask], 
                  c=[colors[cluster_id]], 
                  label=f'Cluster {cluster_id}',
                  alpha=0.6, s=20)
    
    ax.set_xlabel(f'PC{pc_x+1} Score')
    ax.set_ylabel(f'PC{pc_y+1} Score')
    ax.set_title(f'Cluster Scatter: PC{pc_x+1} vs PC{pc_y+1}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_fig(fig, output_dir / f"{base_name}_PC{pc_x+1}_PC{pc_y+1}.png")


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    output_dir: Path,
    base_name: str = "dendrogram",
    method: str = "ward"
):
    """
    Plot hierarchical clustering dendrogram.
    
    Args:
        linkage_matrix: Linkage matrix from scipy.cluster.hierarchy
        output_dir: Directory to save plot
        base_name: Base filename for output
        method: Linkage method name for title
    """
    try:
        from scipy.cluster.hierarchy import dendrogram
    except ImportError:
        logger.warning("scipy not available for dendrogram plotting")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dendrogram(linkage_matrix, ax=ax)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Distance')
    ax.set_title(f'Hierarchical Clustering Dendrogram ({method})')
    
    save_fig(fig, output_dir / f"{base_name}_{method}.png")


def plot_cluster_size_distribution(
    cluster_info: list,
    output_dir: Path,
    base_name: str = "cluster_sizes"
):
    """
    Plot bar chart of cluster sizes.
    
    Args:
        cluster_info: List of dicts with 'cluster' and 'size'
        output_dir: Directory to save plot
        base_name: Base filename for output
    """
    clusters = [info['cluster'] for info in cluster_info if info['size'] > 0]
    sizes = [info['size'] for info in cluster_info if info['size'] > 0]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    ax.bar(clusters, sizes, color=colors, edgecolor='black')
    
    ax.set_xlabel('Cluster ID', fontsize=15)
    ax.set_ylabel('Number of Pixels', fontsize=15)
    ax.set_title('Cluster Size Distribution', fontsize=16)
    ax.set_xticks(clusters)
    ax.set_xticklabels(clusters, fontsize=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    total_pixels = sum(sizes)
    for i, (cluster, size) in enumerate(zip(clusters, sizes)):
        percentage = (size / total_pixels) * 100
        ax.text(cluster, size, f'{percentage:.1f}%', 
               ha='center', va='bottom', fontweight='bold')
    
    save_fig(fig, output_dir / f"{base_name}.png")
