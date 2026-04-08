"""
Plot Utilities Module

Common utilities for plot configuration, styling, and file operations
shared across all plotting modules.
"""

import yaml
import re
import matplotlib.pyplot as plt
from pathlib import Path

# Default plot configuration
DEFAULT_PLOT_CONFIG = {
    'plot_settings': {
        'figure_sizes': {
            'single_plot': [10, 8],
            'summary_plot': [16, 12],
            'comparison_plot': [10, 8],
            'small_plot': [8, 6],
            'single_stacked': [6, 8],
            'multi_stacked_base': [6, 8]
        },
        'dpi': 300,
        'fonts': {
            'title_size': 18,
            'subtitle_size': 14,
            'axis_label_size': 14,
            'tick_label_size': 14,
            'legend_size': 14,
            'info_text_size': 12
        },
        'colors': {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'background': '#F5F5F5'
        },
        'lines': {
            'linewidth': 3.0,
            'grid_alpha': 0.3,
            'marker_size': 8
        },
        'legends': {
            'location': 'best',
            'frameon': True,
            'fancybox': True,
            'shadow': True
        }
    },
    'export': {
        'default_format': 'png',
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'transparent': False
    }
}


def load_plot_config(config_path=None):
    """
    Load plot configuration from YAML file with fallback to defaults.

    Args:
        config_path (Path, optional): Path to plot_settings.yaml file

    Returns:
        dict: Complete plot configuration
    """
    # Find config file if not provided
    if config_path is None:
        current_dir = Path(__file__).parent
        # Look for config in various possible locations
        search_paths = [
            current_dir / "../../../../project_root/xps_config/plot_settings.yaml",
            current_dir / "../../../project_root/xps_config/plot_settings.yaml",
            current_dir / "../../xps_config/plot_settings.yaml", 
            current_dir / "../plot_settings.yaml",
            current_dir / "plot_settings.yaml"
        ]

        for path in search_paths:
            if path.exists():
                config_path = path.resolve()
                break

    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)

            # Merge user config with defaults (user config takes precedence)
            def merge_dicts(default, user):
                result = default.copy()
                for key, value in user.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = merge_dicts(result[key], value)
                    else:
                        result[key] = value
                return result

            config = merge_dicts(DEFAULT_PLOT_CONFIG, user_config)
            print(f"Loaded plot configuration from: {config_path}")
            return config

        except Exception as e:
            print(f"Error loading plot config from {config_path}: {e}")
            print("   Using default settings")
    else:
        print("Using default plot settings (no config file found)")

    return DEFAULT_PLOT_CONFIG


def save_figure_with_config(fig, output_paths, config=None, dpi_override=None):
    """
    Save figure to multiple paths using configuration settings.

    Args:
        fig: Matplotlib figure object
        output_paths (list): List of paths to save figure to
        config (dict, optional): Plot configuration dict
        dpi_override (int, optional): Override DPI setting

    Returns:
        list: List of paths where figure was saved
    """
    if config is None:
        config = load_plot_config()

    export_config = config['export']
    dpi_to_use = dpi_override if dpi_override else config['plot_settings']['dpi']

    saved_paths = []

    for output_path in output_paths:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            fig.savefig(
                output_path,
                dpi=dpi_to_use,
                bbox_inches=export_config['bbox_inches'],
                facecolor=export_config['facecolor'],
                transparent=export_config['transparent']
            )
            saved_paths.append(output_path)
        except Exception as e:
            print(f"  Error saving plot to {output_path}: {e}")

    return saved_paths


def get_plot_colors(n_colors, colormap='viridis', config=None):
    """
    Get a list of colors for plotting multiple data series.

    Args:
        n_colors (int): Number of colors needed
        colormap (str): Matplotlib colormap name
        config (dict, optional): Plot configuration

    Returns:
        list: List of color values
    """
    import matplotlib.cm as cm
    import numpy as np

    if n_colors <= 0:
        return []
    elif n_colors == 1:
        if config:
            return [config['plot_settings']['colors']['primary']]
        else:
            return ['#2E86AB']
    else:
        cmap = cm.get_cmap(colormap)
        return [cmap(i / (n_colors - 1)) for i in range(n_colors)]


def sanitize_filename(filename):
    """
    Sanitize filename for safe filesystem use.

    Args:
        filename (str): Input filename

    Returns:
        str: Sanitized filename
    """
    return re.sub(r'[<>:"/\\|?*]', '_', str(filename))


def setup_plot_style(config=None):
    """
    Apply plot styling from configuration.

    Args:
        config (dict, optional): Plot configuration
    """
    if config is None:
        config = load_plot_config()

    # Set matplotlib rcParams for consistent styling
    font_config = config['plot_settings']['fonts']

    plt.rcParams.update({
        'font.size': font_config['tick_label_size'],
        'axes.titlesize': font_config['title_size'],
        'axes.labelsize': font_config['axis_label_size'],
        'xtick.labelsize': font_config['tick_label_size'],
        'ytick.labelsize': font_config['tick_label_size'],
        'legend.fontsize': font_config['legend_size'],
        'figure.dpi': config['plot_settings']['dpi'],
        'savefig.dpi': config['plot_settings']['dpi'],
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white'
    })


def create_subplot_layout(n_plots, max_cols=3):
    """
    Calculate optimal subplot layout for given number of plots.

    Args:
        n_plots (int): Number of plots needed
        max_cols (int): Maximum number of columns

    Returns:
        tuple: (rows, cols) for subplot layout
    """
    if n_plots <= 0:
        return (1, 1)
    elif n_plots <= max_cols:
        return (1, n_plots)
    else:
        cols = max_cols
        rows = (n_plots + cols - 1) // cols  # Ceiling division
        return (rows, cols)
