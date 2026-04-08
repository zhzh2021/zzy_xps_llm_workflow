"""
Correlation Analysis Plots Module

Functions for visualizing correlations between different XPS parameters,
regions, and samples.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr

# Import utilities - handle both relative and absolute imports
try:
    from plot_modules.utils.plot_utils import load_plot_config
except ImportError:
    try:
        from ..utils.plot_utils import load_plot_config
    except ImportError:
        # Fallback for standalone usage
        import sys
        from pathlib import Path
        utils_path = Path(__file__).parent.parent / "utils"
        sys.path.insert(0, str(utils_path))
        from plot_utils import load_plot_config


def plot_correlation_matrix(data_df, output_dir, title="XPS Parameter Correlations", config=None):
    """
    Create correlation matrix heatmap for XPS data parameters.
    
    Args:
        data_df (pd.DataFrame): DataFrame with numerical columns to correlate
        output_dir (Path): Directory to save the plot
        title (str): Title for the correlation plot
        config (dict, optional): Plot configuration
        
    Returns:
        Path: Path to saved correlation plot
    """
    if config is None:
        config = load_plot_config()
    
    plot_config = config['plot_settings']
    
    # Calculate correlation matrix
    correlation_matrix = data_df.corr()
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=tuple(plot_config['figure_sizes']['comparison_plot']))
    
    # Create heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
    
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        center=0,
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation Coefficient'},
        ax=ax
    )
    
    font_config = plot_config['fonts']
    ax.set_title(title, fontsize=font_config['title_size'], fontweight='bold')
    ax.set_xlabel('Parameters', fontsize=font_config['axis_label_size'], fontweight='bold')
    ax.set_ylabel('Parameters', fontsize=font_config['axis_label_size'], fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / f'correlation_matrix.{config["export"]["default_format"]}'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(
        output_path,
        dpi=plot_config['dpi'],
        bbox_inches=config['export']['bbox_inches'],
        facecolor=config['export']['facecolor']
    )
    plt.close(fig)
    
    return output_path


def plot_scatter_correlation(x_data, y_data, x_label, y_label, output_dir, 
                           sample_labels=None, config=None):
    """
    Create scatter plot showing correlation between two parameters.
    
    Args:
        x_data (array-like): X-axis data
        y_data (array-like): Y-axis data  
        x_label (str): Label for X-axis
        y_label (str): Label for Y-axis
        output_dir (Path): Directory to save the plot
        sample_labels (list, optional): Labels for each data point
        config (dict, optional): Plot configuration
        
    Returns:
        Path: Path to saved scatter plot
    """
    if config is None:
        config = load_plot_config()
    
    plot_config = config['plot_settings']
    
    fig, ax = plt.subplots(figsize=tuple(plot_config['figure_sizes']['single_plot']))
    
    # Create scatter plot
    scatter = ax.scatter(
        x_data, 
        y_data, 
        alpha=0.7,
        s=plot_config['lines']['marker_size'] * 10,
        c=plot_config['colors']['primary']
    )
    
    # Add sample labels if provided
    if sample_labels:
        for i, label in enumerate(sample_labels):
            ax.annotate(label, (x_data[i], y_data[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=plot_config['fonts']['info_text_size'])
    
    # Calculate and display correlation
    if len(x_data) > 1 and len(y_data) > 1:
        corr_coeff, p_value = pearsonr(x_data, y_data)
        
        # Add trend line
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        ax.plot(x_data, p(x_data), 
               color=plot_config['colors']['secondary'],
               linestyle='--', alpha=0.8)
        
        # Add correlation info
        ax.text(0.05, 0.95, f'r = {corr_coeff:.3f}\np = {p_value:.3f}',
               transform=ax.transAxes,
               fontsize=plot_config['fonts']['info_text_size'],
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    font_config = plot_config['fonts']
    ax.set_xlabel(x_label, fontsize=font_config['axis_label_size'], fontweight='bold')
    ax.set_ylabel(y_label, fontsize=font_config['axis_label_size'], fontweight='bold')
    ax.set_title(f'{y_label} vs {x_label}', fontsize=font_config['title_size'], fontweight='bold')
    ax.grid(True, alpha=plot_config['lines']['grid_alpha'])
    
    plt.tight_layout()
    
    # Save plot
    safe_x_label = x_label.replace(' ', '_').replace('/', '_')
    safe_y_label = y_label.replace(' ', '_').replace('/', '_')
    output_path = Path(output_dir) / f'{safe_y_label}_vs_{safe_x_label}.{config["export"]["default_format"]}'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(
        output_path,
        dpi=plot_config['dpi'],
        bbox_inches=config['export']['bbox_inches'],
        facecolor=config['export']['facecolor']
    )
    plt.close(fig)
    
    return output_path


def plot_multi_parameter_correlation(data_df, parameter_columns, output_dir, config=None):
    """
    Create pairwise correlation plots for multiple parameters.
    
    Args:
        data_df (pd.DataFrame): DataFrame containing the parameters
        parameter_columns (list): List of column names to correlate
        output_dir (Path): Directory to save the plot
        config (dict, optional): Plot configuration
        
    Returns:
        Path: Path to saved multi-correlation plot
    """
    if config is None:
        config = load_plot_config()
    
    plot_config = config['plot_settings']
    
    # Select only the specified parameters
    subset_df = data_df[parameter_columns].dropna()
    
    if subset_df.empty or len(parameter_columns) < 2:
        print("Insufficient data for multi-parameter correlation")
        return None
    
    # Create pairplot
    fig = plt.figure(figsize=tuple(plot_config['figure_sizes']['summary_plot']))
    
    # Use seaborn's pairplot for comprehensive correlation visualization
    g = sns.pairplot(
        subset_df,
        diag_kind='hist',
        plot_kws={
            'alpha': 0.7,
            's': plot_config['lines']['marker_size'] * 5
        }
    )
    
    # Style the plot
    font_config = plot_config['fonts']
    g.fig.suptitle('Multi-Parameter Correlation Analysis', 
                   fontsize=font_config['title_size'], fontweight='bold',
                   y=1.02)
    
    # Save plot
    output_path = Path(output_dir) / f'multi_parameter_correlation.{config["export"]["default_format"]}'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    g.savefig(
        output_path,
        dpi=plot_config['dpi'],
        bbox_inches=config['export']['bbox_inches'],
        facecolor=config['export']['facecolor']
    )
    plt.close(g.fig)
    
    return output_path


def plot_region_comparison(region_data_dict, output_dir, config=None):
    """
    Create comparative plots between different XPS regions.
    
    Args:
        region_data_dict (dict): Dictionary with region names as keys and data as values
        output_dir (Path): Directory to save the plot
        config (dict, optional): Plot configuration
        
    Returns:
        Path: Path to saved region comparison plot
    """
    if config is None:
        config = load_plot_config()
    
    plot_config = config['plot_settings']
    
    if len(region_data_dict) < 2:
        print("Need at least 2 regions for comparison")
        return None
    
    # Create comparison figure
    n_regions = len(region_data_dict)
    n_cols = min(3, n_regions)
    n_rows = (n_regions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=tuple(plot_config['figure_sizes']['summary_plot']),
        squeeze=False
    )
    
    font_config = plot_config['fonts']
    
    for idx, (region_name, data) in enumerate(region_data_dict.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Plot region-specific data (example: intensity histogram)
        if isinstance(data, (list, np.ndarray, pd.Series)):
            ax.hist(data, bins=20, alpha=0.7, 
                   color=plot_config['colors']['primary'])
            ax.set_title(f'{region_name}', 
                        fontsize=font_config['subtitle_size'], fontweight='bold')
            ax.set_xlabel('Intensity', fontsize=font_config['axis_label_size'])
            ax.set_ylabel('Frequency', fontsize=font_config['axis_label_size'])
        
        ax.grid(True, alpha=plot_config['lines']['grid_alpha'])
    
    # Hide empty subplots
    for idx in range(n_regions, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle('XPS Region Comparison', 
                fontsize=font_config['title_size'], fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / f'region_comparison.{config["export"]["default_format"]}'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(
        output_path,
        dpi=plot_config['dpi'],
        bbox_inches=config['export']['bbox_inches'],
        facecolor=config['export']['facecolor']
    )
    plt.close(fig)
    
    return output_path