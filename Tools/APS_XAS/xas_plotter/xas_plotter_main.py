"""
XAS Data Plotter Module - Main Entry Point

Central hub for all XAS data visualization tools.
Imports and orchestrates calls to specialized plotting modules:
- Raw data plotting (xas_rawdata_plotter)
- Quality control plotting (quality_control_plotter)
- Feature comparison plotting (xas_features_plotter)
- Quality report plotting (xas_quality_report_plots)

Creates publication-quality plots for XANES, EXAFS, and Fourier transforms.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import matplotlib.gridspec as gridspec
import yaml

# Import specialized plotting modules
try:
    from .xas_rawdata_plotter import XASPlotter as XASRawDataPlotter
except ImportError:
    try:
        from xas_rawdata_plotter import XASPlotter as XASRawDataPlotter
    except ImportError:
        XASRawDataPlotter = None

try:
    from .quality_control_plotter import XASQualityControlPlotter
except ImportError:
    try:
        from quality_control_plotter import XASQualityControlPlotter
    except ImportError:
        XASQualityControlPlotter = None

try:
    from .xas_features_plotter import create_feature_comparison_plots as create_features_plots
except ImportError:
    try:
        from xas_features_plotter import create_feature_comparison_plots as create_features_plots
    except ImportError:
        create_features_plots = None

try:
    from .xas_quality_report_plots import plot_xas_quality_report_diagnostics
except ImportError:
    try:
        from xas_quality_report_plots import plot_xas_quality_report_diagnostics
    except ImportError:
        plot_xas_quality_report_diagnostics = None


class XASPlotter:
    """
    Main XAS data visualization orchestrator.

    Central hub that coordinates all plotting functionality:
    - Raw data visualization
    - Quality control diagnostics
    - Feature comparisons
    - Quality reports

    Delegates specific plotting tasks to specialized modules.
    """

    def __init__(self,
                 settings_file: Optional[str | Path] = None,
                 figsize: Optional[Tuple[int, int]] = None,
                 dpi: Optional[int] = None,
                 style: Optional[str] = None):
        """
        Initialize main plotter with settings and specialized plotters.

        Parameters
        ----------
        settings_file : str or Path, optional
            Path to YAML settings file. If None, uses default.
        figsize : tuple, optional
            Default figure size (width, height). Overrides settings file.
        dpi : int, optional
            Resolution for saved figures. Overrides settings file.
        style : str, optional
            Matplotlib style. Overrides settings file.
        """
        # Load settings from YAML file
        self.settings = self._load_plot_settings(settings_file)

        # Override with provided parameters if given
        if figsize is not None:
            self.figsize = figsize
        else:
            self.figsize = tuple(self.settings['plot_settings']['figure_sizes']['xanes_single'])

        if dpi is not None:
            self.dpi = dpi
        else:
            self.dpi = self.settings['plot_settings']['dpi']

        if style is not None:
            self.style = style
        else:
            self.style = self.settings['advanced']['matplotlib_style']

        # Apply publication quality if enabled
        if self.settings.get('use_publication_quality', False):
            self._apply_publication_settings()

        # Set matplotlib style
        plt.style.use(self.style)

        # Initialize specialized plotters
        self._init_specialized_plotters()

        # Default colors for multi-sample plots
        self.colors = plt.cm.tab10.colors

    def _init_specialized_plotters(self):
        """Initialize specialized plotting modules."""
        if XASRawDataPlotter is not None:
            try:
                self.raw_plotter = XASRawDataPlotter(
                    settings_file=self._get_settings_file(),
                    figsize=self.figsize,
                    dpi=self.dpi,
                    style=self.style
                )
            except Exception as e:
                print(f"Warning: Could not initialize raw data plotter: {e}")
                self.raw_plotter = None
        else:
            print("Warning: XASRawDataPlotter not available")
            self.raw_plotter = None

        if XASQualityControlPlotter is not None:
            try:
                self.quality_plotter = XASQualityControlPlotter(
                    plot_settings=self.settings.get('plot_settings'),
                    save_path=None
                )
            except Exception as e:
                print(f"Warning: Could not initialize quality control plotter: {e}")
                self.quality_plotter = None
        else:
            print("Warning: XASQualityControlPlotter not available")
            self.quality_plotter = None

    def _get_settings_file(self) -> Optional[Path]:
        """Get the path to the settings file."""
        module_dir = Path(__file__).parent
        settings_file = module_dir / "xas_plot_settings.yaml"
        if settings_file.exists():
            return settings_file
        return None

    def _load_plot_settings(self, settings_file: Optional[str | Path] = None) -> Dict[str, Any]:
        """
        Load plot settings from YAML file.

        Parameters
        ----------
        settings_file : str or Path, optional
            Path to YAML settings file

        Returns
        -------
        settings : dict
            Plot settings dictionary
        """
        if settings_file is None:
            # Use default settings file in the same directory as this module
            module_dir = Path(__file__).parent
            settings_file = module_dir / "xas_plot_settings.yaml"

        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = yaml.safe_load(f)
            print(f"Loaded XAS plot settings from {settings_file}")
            return settings
        except FileNotFoundError:
            print(f"Warning: Settings file {settings_file} not found, using defaults")
            return self._get_default_settings()
        except Exception as e:
            print(f"Warning: Error loading settings file {settings_file}: {e}")
            return self._get_default_settings()

    def _apply_publication_settings(self):
        """Apply publication quality settings."""
        pub_settings = self.settings.get('publication', {})

        # Update with publication fonts
        if 'fonts' in pub_settings:
            self.settings['plot_settings']['fonts'].update(pub_settings['fonts'])

        # Update with publication lines
        if 'lines' in pub_settings:
            self.settings['plot_settings']['lines'].update(pub_settings['lines'])

        # Update with publication markers
        if 'markers' in pub_settings:
            self.settings['plot_settings']['markers'].update(pub_settings['markers'])

        # Update with publication legend settings
        if 'legend' in pub_settings:
            self.settings['plot_settings']['legend'].update(pub_settings['legend'])

        # Update DPI for publication quality
        if 'dpi' in pub_settings:
            self.dpi = pub_settings['dpi']

        # Update figure sizes to publication sizes
        if 'figure_sizes' in pub_settings:
            # Override default figure sizes with publication sizes
            self.settings['plot_settings']['figure_sizes'].update(pub_settings['figure_sizes'])

        # Update DPI
        if 'dpi' in pub_settings:
            self.dpi = pub_settings['dpi']

    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default plot settings when YAML file is not available."""
        return {
            'plot_settings': {
                'figure_sizes': {
                    'xanes_single': [12, 8],
                    'exafs_single': [12, 8],
                    'ft_single': [12, 8],
                    'combined_single': [12, 10],
                    'comparison_xanes': [12, 8],
                    'comparison_exafs': [12, 8],
                    'comparison_ft': [12, 8],
                    'overview_panel': [16, 12]
                },
                'dpi': 300,
                'fonts': {
                    'title_size': 20,
                    'subtitle_size': 18,
                    'axis_label_size': 18,
                    'tick_label_size': 16,
                    'legend_size': 18,
                    'annotation_size': 16
                },
                'lines': {
                    'data_line_width': 3.5,
                    'fit_line_width': 3.0,
                    'grid_line_width': 1.2,
                    'axis_line_width': 1.8
                },
                'markers': {
                    'size': 8,
                    'edge_width': 1.5
                }
            },
            'advanced': {
                'matplotlib_style': 'seaborn-v0_8-whitegrid'
            },
            'use_publication_quality': False
        }

    # ===== DELEGATION METHODS TO SPECIALIZED PLOTTERS =====

    def plot_raw_data(self, results: Dict[str, Any],
                     save_path: Optional[str | Path] = None,
                     show_plot: bool = True) -> plt.Figure:
        """
        Plot raw XAS data using the raw data plotter.

        Parameters
        ----------
        results : dict
            Results from XASProcessor.process_single_spectrum
        save_path : str or Path, optional
            Path to save the figure
        show_plot : bool
            Whether to display the plot

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        """
        if self.raw_plotter is None:
            raise RuntimeError("Raw data plotter not available")
        return self.raw_plotter.plot_raw_data(results, save_path, show_plot)

    def plot_quality_control(self, results: Dict[str, Any],
                           save_path: Optional[str | Path] = None,
                           show_plot: bool = True) -> plt.Figure:
        """
        Plot quality control diagnostics using the quality control plotter.

        Parameters
        ----------
        results : dict
            Results from XASProcessor.process_single_spectrum
        save_path : str or Path, optional
            Path to save the figure
        show_plot : bool
            Whether to display the plot

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        """
        if self.quality_plotter is None:
            raise RuntimeError("Quality control plotter not available")
        return self.quality_plotter.plot_quality_control(results, save_path, show_plot)

    def plot_feature_comparison(self, batch_results: Dict[str, Any],
                               save_path: Optional[str | Path] = None,
                               show_plot: bool = True) -> plt.Figure:
        """
        Plot feature comparison across samples using the features plotter.

        Parameters
        ----------
        batch_results : dict
            Batch results from XASProcessor.process_batch
        save_path : str or Path, optional
            Path to save the figure
        show_plot : bool
            Whether to display the plot

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        """
        if create_features_plots is None:
            raise RuntimeError("Feature comparison plotter not available")
        return create_features_plots(batch_results, save_path, show_plot)

    def plot_quality_report(self, results: Dict[str, Any],
                           save_path: Optional[str | Path] = None,
                           show_plot: bool = True) -> plt.Figure:
        """
        Plot quality report diagnostics using the quality report plotter.

        Parameters
        ----------
        results : dict
            Results from XASProcessor.process_single_spectrum
        save_path : str or Path, optional
            Path to save the figure
        show_plot : bool
            Whether to display the plot

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        """
        if plot_xas_quality_report_diagnostics is None:
            raise RuntimeError("Quality report plotter not available")
        return plot_xas_quality_report_diagnostics(results, save_path, show_plot)

    # ===== CONVENIENCE METHODS FOR COMMON WORKFLOWS =====

    def plot_complete_analysis(self, results: Dict[str, Any],
                              output_dir: str | Path,
                              show_plot: bool = False) -> Dict[str, str]:
        """
        Create complete set of plots for a single sample analysis.

        Parameters
        ----------
        results : dict
            Results from XASProcessor.process_single_spectrum
        output_dir : str or Path
            Directory to save all plots
        show_plot : bool
            Whether to display plots (default False for batch processing)

        Returns
        -------
        plot_files : dict
            Dictionary mapping plot types to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        plot_files = {}

        # Raw data plot
        if self.raw_plotter is not None:
            raw_path = output_dir / f"{results['sample_name']}_raw_data.png"
            self.plot_raw_data(results, raw_path, show_plot)
            plot_files['raw_data'] = str(raw_path)

        # Quality control plot
        if self.quality_plotter is not None:
            qc_path = output_dir / f"{results['sample_name']}_quality_control.png"
            self.plot_quality_control(results, qc_path, show_plot)
            plot_files['quality_control'] = str(qc_path)

        # Quality report plot
        if plot_xas_quality_report_diagnostics is not None:
            report_path = output_dir / f"{results['sample_name']}_quality_report.png"
            self.plot_quality_report(results, report_path, show_plot)
            plot_files['quality_report'] = str(report_path)

        return plot_files

    def plot_batch_analysis(self, batch_results: Dict[str, Any],
                           output_dir: str | Path,
                           show_plot: bool = False) -> Dict[str, str]:
        """
        Create complete set of plots for batch analysis results.

        Parameters
        ----------
        batch_results : dict
            Batch results from XASProcessor.process_batch
        output_dir : str or Path
            Directory to save all plots
        show_plot : bool
            Whether to display plots (default False for batch processing)

        Returns
        -------
        plot_files : dict
            Dictionary mapping plot types to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        plot_files = {}

        # Feature comparison plot
        if create_features_plots is not None:
            features_path = output_dir / "feature_comparison.png"
            self.plot_feature_comparison(batch_results, features_path, show_plot)
            plot_files['feature_comparison'] = str(features_path)

        # Individual sample plots
        for sample_name, results in batch_results.items():
            sample_dir = output_dir / sample_name
            sample_plots = self.plot_complete_analysis(results, sample_dir, show_plot)
            plot_files.update({f"{sample_name}_{k}": v for k, v in sample_plots.items()})

        return plot_files

    