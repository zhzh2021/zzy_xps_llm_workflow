"""
XPS Plotter Modules

Modular plotting system for XPS workflow visualization.
Each module handles specific types of plots for different workflow stages.
"""

# Import key plotting functions from submodules for easy access (with error handling)
try:
    from .data_quality.quality_report_plots import plot_quality_report_diagnostics, plot_sample_quality_details
except ImportError:
    plot_quality_report_diagnostics = None
    plot_sample_quality_details = None

try:
    from .fitting.fitting_plots import plot_template_fit, plot_stacked_layers_comparison
except ImportError:
    plot_template_fit = None
    plot_stacked_layers_comparison = None

try:
    from .quantification.quantification_plots import (
        plot_atomic_concentration_per_sample,
        plot_atomic_concentration_layer_comparison,
        plot_chemistry_heatmap,
        plot_quantification_overview,
        create_quantification_plots,
        generate_component_chemistry_plots,
        generate_component_heatmap,
    )
except ImportError:
    plot_atomic_concentration_per_sample = None
    plot_atomic_concentration_layer_comparison = None
    plot_chemistry_heatmap = None
    plot_quantification_overview = None
    create_quantification_plots = None
    generate_component_chemistry_plots = None
    generate_component_heatmap = None

try:
    from .correlation.correlation_plots import plot_correlation_matrix
except ImportError:
    plot_correlation_matrix = None

try:
    from .utils.plot_utils import load_plot_config, save_figure_with_config
except ImportError:
    load_plot_config = None
    save_figure_with_config = None

__version__ = "1.0.0"

__all__ = [
    'plot_extracted_region_quality',
    'plot_all_extracted_regions_quality',
    'plot_template_fit', 
    'plot_stacked_layers_comparison',
    'plot_atomic_concentration_per_sample',
    'plot_atomic_concentration_layer_comparison',
    'plot_chemistry_heatmap',
    'plot_quantification_overview',
    'create_quantification_plots',
    'generate_component_chemistry_plots',
    'generate_component_heatmap',
    'plot_correlation_matrix',
    'load_plot_config',
    'save_figure_with_config'
]
