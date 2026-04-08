"""
3D Waterfall Plot Module for XPS Depth Profile Visualization

Provides function to generate 3D waterfall plots for XPS depth profiles with multiple layers.
Each spectrum layer is plotted on a single 3D axes with depth-coded colors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, Optional, Tuple


def plot_depth_profile_3d_waterfall(
    spectra_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    region: str,
    out_dir: Path,
    cmap_name: str = "viridis",
    config=None,
    depth_direction: str = "bulk_to_surface",
) -> Optional[Path]:
    """
    Create a 3D waterfall plot for depth profile visualization.
    
    Visualizes all layers of an XPS region on a single 3D plot:
    - X-axis: Binding Energy (eV)
    - Y-axis: Layer number (depth, surface→bulk)
    - Z-axis: Intensity (cps)
    - Colors: Progressive from surface (blue) to bulk (yellow)
    - Vertical drops: Show intensity variation at each layer
    
    Useful for:
    - Identifying peak shifts across depth
    - Visualizing intensity trends
    - Comparing surface vs bulk chemistry
    - Interactive 3D exploration (user can rotate plot)
    
    Args:
        spectra_dict: Dictionary with layer indices as keys, 
                     (energy array, intensity array) tuples as values
        region: Region name (e.g., "F1s", "C1s")
        out_dir: Output directory for the plot
        cmap_name: Colormap name ('viridis', 'plasma', 'coolwarm', etc.)
        config: Plot configuration dictionary (optional)
        
    Returns:
        Path to saved figure, or None if failed/insufficient data
        
    Example:
        >>> layers = {
        ...     1: (energy_arr1, intensity_arr1),
        ...     2: (energy_arr2, intensity_arr2),
        ...     3: (energy_arr3, intensity_arr3),
        ... }
        >>> path = plot_depth_profile_3d_waterfall(layers, "F1s", Path("plots"))
    """
    
    # Validation
    if not spectra_dict or len(spectra_dict) < 2:
        return None
    
    depth_direction = (depth_direction or "bulk_to_surface").strip().lower()
    if depth_direction in ("surface_to_bulk", "surface->bulk", "surface_to_bulk"):
        sorted_layers = sorted(spectra_dict.keys())
        depth_label = "Depth_Layer"
        depth_title = "Surface -> Bulk"
    else:
        sorted_layers = sorted(spectra_dict.keys(), reverse=True)
        depth_label = "Depth_Layer"
        depth_title = "Bulk -> Surface"
    num_layers = len(sorted_layers)
    
    # Load config if not provided
    if config is None:
        try:
            from plot_modules.utils.plot_utils import load_plot_config
            config = load_plot_config()
        except ImportError:
            # Fallback config
            config = {
                'plot_settings': {
                    'dpi': 300,
                    'figure_sizes': {'summary_plot': (14, 9)},
                    'fonts': {
                        'title_size': 14,
                        'axis_label_size': 12,
                        'legend_size': 12,
                    }
                },
                'export': {
                    'default_format': 'png',
                    'bbox_inches': 'tight',
                    'facecolor': 'white'
                }
            }
    
    # Setup colormap
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, num_layers))
    
    # Create 3D figure
    plot_config = config['plot_settings']
    fig = plt.figure(figsize=tuple(plot_config['figure_sizes'].get('summary_plot', (14, 9))))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each layer as a line in 3D space
    for idx, layer_num in enumerate(sorted_layers):
        energy, intensity = spectra_dict[layer_num]
        
        # Create arrays for this layer
        x = energy
        y = np.full_like(energy, layer_num, dtype=float)
        z = intensity
        
        # Plot the main spectrum line
        ax.plot(x, y, z, 
               color=colors[idx], 
               alpha=0.9, 
               linewidth=2.5,
               label=f"Layer {layer_num}")
        
        # Add vertical connections for waterfall effect (sampled for clarity)
        # This creates the "drops" from spectrum to baseline
        for i in range(0, len(energy), max(1, len(energy)//15)):
            ax.plot([energy[i], energy[i]], 
                   [layer_num, layer_num], 
                   [intensity[i], 0],
                   color=colors[idx],
                   alpha=0.15,
                   linewidth=0.8)
    
    # Customize axes
    font_cfg = plot_config['fonts']
    ax.set_xlabel("Binding Energy (eV)", fontsize=font_cfg['axis_label_size'], labelpad=14)
    ax.set_ylabel(depth_label, fontsize=font_cfg['axis_label_size'], labelpad=14)
    ax.set_zlabel("Intensity (cps)", fontsize=font_cfg['axis_label_size'], labelpad=14)
    
    ax.set_title(
        f"3D Waterfall - {region} Depth Profile\n({depth_title}, rotate to explore)",
        fontsize=font_cfg['title_size'],
        fontweight='bold',
        pad=20
    )
    
    # Invert X-axis for XPS convention (binding energy decreases to right)
    ax.invert_xaxis()
    
    # Set viewing angle for good initial visualization
    # User can interactively rotate with mouse after opening
    ax.view_init(elev=20, azim=120)
    
    # Add legend
    ax.legend(
        loc='upper left', 
        fontsize=font_cfg['legend_size'], 
        ncol=min(3, num_layers), 
        framealpha=0.95
    )
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    try:
        export_cfg = config['export']
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize region name for filename
        region_safe = region.replace('/', '_').replace('\\', '_')
        fname = out_dir / f"depth_profile_3d_waterfall_{region_safe}.{export_cfg['default_format']}"
        
        fig.savefig(
            fname, 
            dpi=plot_config['dpi'], 
            bbox_inches=export_cfg['bbox_inches'],
            facecolor=export_cfg['facecolor']
        )
        plt.close(fig)
        
        return fname
    except Exception as e:
        print(f"⚠️  Error saving 3D waterfall plot: {e}")
        plt.close(fig)
        return None


def load_depth_profile_csv(csv_path: Path) -> Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """
    Load depth profile CSV and extract layer-specific spectra.
    
    Expects CSV format:
    - First column: Binding Energy
    - Other columns: {SampleName}_L{LayerNum}_{Region}_cps
    
    Args:
        csv_path: Path to aggregated CSV file
        
    Returns:
        Dictionary {layer_num: (energy_array, intensity_array)} or None if failed
    """
    
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path, comment='#')
        
        if df.empty or len(df.columns) < 2:
            return None
        
        # Extract energy (first column)
        energy_col = df.columns[0]
        energy = df[energy_col].values
        
        # Parse layer columns
        import re
        layers = {}
        
        for col in df.columns[1:]:
            match = re.search(r'_L(\d+)_', col)
            if match:
                layer = int(match.group(1))
                intensity = df[col].values
                layers[layer] = (energy, intensity)
        
        return layers if layers else None
    except Exception as e:
        print(f"⚠️  Error loading depth profile CSV: {e}")
        return None


if __name__ == "__main__":
    # Test example
    print("3D Waterfall Plot Module")
    print("=" * 60)
    print("\nUsage:")
    print("  from depth_3d_waterfall import plot_depth_profile_3d_waterfall")
    print("  layers = load_depth_profile_csv(Path('aggregated_F1s_allHR.csv'))")
    print("  plot_depth_profile_3d_waterfall(layers, 'F1s', Path('plots'))")
