
"""this module provides XAS data visualization for raw data and processed data (i.e. calibrated and normalized  ). """

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import matplotlib.gridspec as gridspec
import yaml


class XASPlotter:
    """
    XAS data visualization tools.

    Creates plots for:
    - XANES spectra (normalized absorption)
    - EXAFS χ(k) data
    - Fourier transforms χ(R)
    - Multi-sample comparisons
    """

    def __init__(self,
                 settings_file: Optional[str | Path] = None,
                 figsize: Optional[Tuple[int, int]] = None,
                 dpi: Optional[int] = None,
                 style: Optional[str] = None):
        """
        Initialize plotter with settings from YAML file.

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

        # Default colors for multi-sample plots
        self.colors = plt.cm.tab10.colors

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

    def plot_xanes(self,
                  results: Dict[str, Any],
                  save_path: Optional[str | Path] = None,
                  show_plot: bool = True) -> plt.Figure:
        """
        Plot normalized XANES spectrum.

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
        # Get settings for this plot type
        plot_settings = self.settings['plot_settings']
        xanes_settings = self.settings['xas_plots']['xanes']

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        data = results['processed_data']
        sample_name = results['sample_name']

        if 'mu_norm' in data and data['mu_norm'] is not None:
            ax.plot(data['energy'], data['mu_norm'],
                   color=plot_settings['colors']['data_line'],
                   linewidth=plot_settings['lines']['data_line_width'],
                   label=sample_name)

            # Apply font settings
            ax.set_xlabel(xanes_settings['x_label'],
                         fontsize=plot_settings['fonts']['axis_label_size'])
            ax.set_ylabel(xanes_settings['y_label'],
                         fontsize=plot_settings['fonts']['axis_label_size'])
            ax.set_title(xanes_settings['title_template'].format(sample_name=sample_name),
                        fontsize=plot_settings['fonts']['title_size'])

            # Apply tick label sizes
            ax.tick_params(axis='both', which='major',
                          labelsize=plot_settings['fonts']['tick_label_size'])

            # Grid settings
            ax.grid(True, alpha=plot_settings['lines']['grid_alpha'] if 'grid_alpha' in plot_settings['lines'] else 0.3)

            # Legend settings
            legend_settings = plot_settings['legend']
            ax.legend(loc=legend_settings['position'],
                     bbox_to_anchor=legend_settings['bbox_anchor'],
                     fontsize=legend_settings['fontsize'],
                     frameon=legend_settings['frameon'])

            # Highlight edge position if available and enabled
            if xanes_settings.get('show_e0_line', True) and 'e0' in results.get('features', {}):
                e0 = results['features']['e0']
                ax.axvline(e0,
                          color=xanes_settings['e0_line_color'],
                          linestyle=xanes_settings['e0_line_style'],
                          linewidth=xanes_settings['e0_line_width'],
                          alpha=0.8,
                          label=xanes_settings['e0_label'].format(e0=e0))
                ax.legend(loc=legend_settings['position'],
                         bbox_to_anchor=legend_settings['bbox_anchor'],
                         fontsize=legend_settings['fontsize'],
                         frameon=legend_settings['frameon'])

        if save_path:
            export_settings = self.settings['export']
            fig.savefig(save_path,
                       dpi=self.dpi,
                       bbox_inches=export_settings['bbox_inches'],
                       facecolor=export_settings['facecolor'],
                       transparent=export_settings['transparent'],
                       pad_inches=export_settings.get('pad_inches', 0.1))
            print(f"Saved XANES plot: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_exafs(self,
                  results: Dict[str, Any],
                  save_path: Optional[str | Path] = None,
                  show_plot: bool = True) -> plt.Figure:
        """
        Plot EXAFS χ(k) data.

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
        # Get settings for this plot type
        plot_settings = self.settings['plot_settings']
        exafs_settings = self.settings['xas_plots']['exafs']

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        data = results['processed_data']
        sample_name = results['sample_name']

        if 'chi' in data and data['chi'] is not None:
            # Apply k-weighting for better visibility
            k_weight = exafs_settings.get('k_weight', 2)
            chi_weighted = data['chi'] * data['k']**k_weight

            ax.plot(data['k'], chi_weighted,
                   color=plot_settings['colors']['data_line'],
                   linewidth=plot_settings['lines']['data_line_width'],
                   label=sample_name)

            # Apply font settings
            ax.set_xlabel(exafs_settings['x_label'],
                         fontsize=plot_settings['fonts']['axis_label_size'])
            ax.set_ylabel(exafs_settings['y_label'],
                         fontsize=plot_settings['fonts']['axis_label_size'])
            ax.set_title(exafs_settings['title_template'].format(sample_name=sample_name),
                        fontsize=plot_settings['fonts']['title_size'])

            # Apply tick label sizes
            ax.tick_params(axis='both', which='major',
                          labelsize=plot_settings['fonts']['tick_label_size'])

            # Grid settings
            ax.grid(True, alpha=plot_settings['lines']['grid_alpha'] if 'grid_alpha' in plot_settings['lines'] else 0.3)

            # Legend settings
            legend_settings = plot_settings['legend']
            ax.legend(loc=legend_settings['position'],
                     bbox_to_anchor=legend_settings['bbox_anchor'],
                     fontsize=legend_settings['fontsize'],
                     frameon=legend_settings['frameon'])

            # Set k-range display if specified
            if 'k_range_display' in exafs_settings:
                k_min, k_max = exafs_settings['k_range_display']
                ax.set_xlim(k_min, k_max)

        if save_path:
            export_settings = self.settings['export']
            fig.savefig(save_path,
                       dpi=self.dpi,
                       bbox_inches=export_settings['bbox_inches'],
                       facecolor=export_settings['facecolor'],
                       transparent=export_settings['transparent'],
                       pad_inches=export_settings.get('pad_inches', 0.1))
            print(f"Saved EXAFS plot: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_fourier_transform(self,
                              results: Dict[str, Any],
                              save_path: Optional[str | Path] = None,
                              show_plot: bool = True) -> plt.Figure:
        """
        Plot Fourier transform χ(R).

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
        # Get settings for this plot type
        plot_settings = self.settings['plot_settings']
        ft_settings = self.settings['xas_plots']['fourier_transform']

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        data = results['processed_data']
        sample_name = results['sample_name']

        if 'chir_mag' in data and data['chir_mag'] is not None:
            ax.plot(data['r'], data['chir_mag'],
                   color=plot_settings['colors']['data_line'],
                   linewidth=plot_settings['lines']['data_line_width'],
                   label=sample_name)

            # Apply font settings
            ax.set_xlabel(ft_settings['x_label'],
                         fontsize=plot_settings['fonts']['axis_label_size'])
            ax.set_ylabel(ft_settings['y_label'],
                         fontsize=plot_settings['fonts']['axis_label_size'])
            ax.set_title(ft_settings['title_template'].format(sample_name=sample_name),
                        fontsize=plot_settings['fonts']['title_size'])

            # Apply tick label sizes
            ax.tick_params(axis='both', which='major',
                          labelsize=plot_settings['fonts']['tick_label_size'])

            # Grid settings
            ax.grid(True, alpha=plot_settings['lines']['grid_alpha'] if 'grid_alpha' in plot_settings['lines'] else 0.3)

            # Legend settings
            legend_settings = plot_settings['legend']
            ax.legend(loc=legend_settings['position'],
                     bbox_to_anchor=legend_settings['bbox_anchor'],
                     fontsize=legend_settings['fontsize'],
                     frameon=legend_settings['frameon'])

            # Set R-range display if specified
            if 'r_range_display' in ft_settings:
                r_min, r_max = ft_settings['r_range_display']
                ax.set_xlim(r_min, r_max)

            # Highlight main peak if available
            if 'ft_peak_r' in results.get('features', {}):
                peak_r = results['features']['ft_peak_r']
                ax.axvline(peak_r,
                          color='red',
                          linestyle='--',
                          linewidth=2.5,
                          alpha=0.8,
                          label=f'Peak at {peak_r:.2f} Å')
                ax.legend(loc=legend_settings['position'],
                         bbox_to_anchor=legend_settings['bbox_anchor'],
                         fontsize=legend_settings['fontsize'],
                         frameon=legend_settings['frameon'])

        if save_path:
            export_settings = self.settings['export']
            fig.savefig(save_path,
                       dpi=self.dpi,
                       bbox_inches=export_settings['bbox_inches'],
                       facecolor=export_settings['facecolor'],
                       transparent=export_settings['transparent'],
                       pad_inches=export_settings.get('pad_inches', 0.1))
            print(f"Saved FT plot: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_combined_spectrum(self,
                              results: Dict[str, Any],
                              save_path: Optional[str | Path] = None,
                              show_plot: bool = True) -> plt.Figure:
        """
        Create a combined plot showing XANES, EXAFS, and FT in subplots.

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
        # Get settings
        plot_settings = self.settings['plot_settings']
        fig_size = plot_settings['figure_sizes'].get('combined_single', [15, 10])
        
        fig = plt.figure(figsize=fig_size, dpi=self.dpi)
        gs = gridspec.GridSpec(2, 2, figure=fig)

        data = results['processed_data']
        sample_name = results['sample_name']

        # XANES plot
        ax1 = fig.add_subplot(gs[0, 0])
        if 'mu_norm' in data and data['mu_norm'] is not None:
            ax1.plot(data['energy'], data['mu_norm'], 
                    color=plot_settings['colors']['data_line'],
                    linewidth=plot_settings['lines']['data_line_width'])
            ax1.set_xlabel('Energy (eV)', fontsize=plot_settings['fonts']['axis_label_size'])
            ax1.set_ylabel('Normalized μ(E)', fontsize=plot_settings['fonts']['axis_label_size'])
            ax1.set_title('XANES', fontsize=plot_settings['fonts']['subtitle_size'])
            ax1.tick_params(axis='both', labelsize=plot_settings['fonts']['tick_label_size'])
            ax1.grid(True, alpha=0.3)

        # EXAFS plot
        ax2 = fig.add_subplot(gs[0, 1])
        if 'chi' in data and data['chi'] is not None:
            ax2.plot(data['k'], data['chi'] * data['k']**2, 
                    color=plot_settings['colors']['fit_line'],
                    linewidth=plot_settings['lines']['data_line_width'])
            ax2.set_xlabel('k (A^-1)', fontsize=plot_settings['fonts']['axis_label_size'])
            ax2.set_ylabel('k²χ(k)', fontsize=plot_settings['fonts']['axis_label_size'])
            ax2.set_title('EXAFS χ(k)', fontsize=plot_settings['fonts']['subtitle_size'])
            ax2.tick_params(axis='both', labelsize=plot_settings['fonts']['tick_label_size'])
            ax2.grid(True, alpha=0.3)

        # Fourier transform
        ax3 = fig.add_subplot(gs[1, :])
        if 'chir_mag' in data and data['chir_mag'] is not None:
            ax3.plot(data['r'], data['chir_mag'], 
                    color='purple',
                    linewidth=plot_settings['lines']['data_line_width'])
            ax3.set_xlabel('R (A)', fontsize=plot_settings['fonts']['axis_label_size'])
            ax3.set_ylabel('|χ(R)|', fontsize=plot_settings['fonts']['axis_label_size'])
            ax3.set_title('Fourier Transform', fontsize=plot_settings['fonts']['subtitle_size'])
            ax3.tick_params(axis='both', labelsize=plot_settings['fonts']['tick_label_size'])
            ax3.grid(True, alpha=0.3)

        fig.suptitle(f'XAS Analysis: {sample_name}', fontsize=plot_settings['fonts']['title_size'])
        plt.tight_layout()

        if save_path:
            export_settings = self.settings['export']
            fig.savefig(save_path, 
                       dpi=self.dpi, 
                       bbox_inches=export_settings['bbox_inches'],
                       facecolor=export_settings['facecolor'],
                       transparent=export_settings['transparent'])
            print(f"Saved combined plot: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_multi_sample_comparison(self,
                                   results_dict: Dict[str, Dict],
                                   plot_type: str = 'xanes',
                                   save_path: Optional[str | Path] = None,
                                   show_plot: bool = True) -> plt.Figure:
        """
        Plot multiple samples for comparison.

        Parameters
        ----------
        results_dict : dict
            Dictionary of results from batch processing
        plot_type : str
            Type of plot: 'xanes', 'exafs', 'ft'
        save_path : str or Path, optional
            Path to save the figure
        show_plot : bool
            Whether to display the plot

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        """
        # Get settings
        plot_settings = self.settings['plot_settings']
        comparison_figsize = plot_settings['figure_sizes'].get(f'comparison_{plot_type}', self.figsize)
        
        fig, ax = plt.subplots(figsize=comparison_figsize, dpi=self.dpi)

        for i, (sample_name, results) in enumerate(results_dict.items()):
            color = self.colors[i % len(self.colors)]
            data = results['processed_data']

            if plot_type == 'xanes' and 'mu_norm' in data and data['mu_norm'] is not None:
                ax.plot(data['energy'], data['mu_norm'],
                       color=color, 
                       linewidth=plot_settings['lines']['data_line_width'], 
                       label=sample_name)
                ax.set_xlabel('Energy (eV)', fontsize=plot_settings['fonts']['axis_label_size'])
                ax.set_ylabel('Normalized μ(E)', fontsize=plot_settings['fonts']['axis_label_size'])
                ax.set_title('XANES Comparison', fontsize=plot_settings['fonts']['title_size'])

            elif plot_type == 'exafs' and 'chi' in data and data['chi'] is not None:
                ax.plot(data['k'], data['chi'] * data['k']**2,
                       color=color, 
                       linewidth=plot_settings['lines']['data_line_width'], 
                       label=sample_name)
                ax.set_xlabel('k (A^-1)', fontsize=plot_settings['fonts']['axis_label_size'])
                ax.set_ylabel('k²χ(k)', fontsize=plot_settings['fonts']['axis_label_size'])
                ax.set_title('EXAFS Comparison', fontsize=plot_settings['fonts']['title_size'])

            elif plot_type == 'ft' and 'chir_mag' in data and data['chir_mag'] is not None:
                ax.plot(data['r'], data['chir_mag'],
                       color=color, 
                       linewidth=plot_settings['lines']['data_line_width'], 
                       label=sample_name)
                ax.set_xlabel('R (A)', fontsize=plot_settings['fonts']['axis_label_size'])
                ax.set_ylabel('|χ(R)|', fontsize=plot_settings['fonts']['axis_label_size'])
                ax.set_title('Fourier Transform Comparison', fontsize=plot_settings['fonts']['title_size'])

        ax.tick_params(axis='both', labelsize=plot_settings['fonts']['tick_label_size'])
        ax.grid(True, alpha=0.3)
        
        # Legend settings
        legend_settings = plot_settings['legend']
        ax.legend(loc=legend_settings['position'],
                 fontsize=legend_settings['fontsize'],
                 frameon=legend_settings['frameon'])
        
        plt.tight_layout()

        if save_path:
            export_settings = self.settings['export']
            fig.savefig(save_path, 
                       dpi=self.dpi, 
                       bbox_inches=export_settings['bbox_inches'],
                       facecolor=export_settings['facecolor'],
                       transparent=export_settings['transparent'])
            print(f"Saved comparison plot: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def create_summary_plots(self,
                           results_dict: Dict[str, Dict],
                           output_dir: str | Path) -> Dict[str, str]:
        """
        Create a complete set of plots for batch analysis results.

        Parameters
        ----------
        results_dict : dict
            Results from batch processing
        output_dir : str or Path
            Directory to save plots

        Returns
        -------
        plot_files : dict
            Dictionary mapping plot types to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        plot_files = {}

        # Individual sample plots
        for sample_name, results in results_dict.items():
            sample_dir = output_dir / sample_name
            sample_dir.mkdir(exist_ok=True)

            # Combined spectrum plot
            combined_path = sample_dir / f"{sample_name}_combined.png"
            self.plot_combined_spectrum(results, combined_path, show_plot=False)
            plot_files[f'{sample_name}_combined'] = str(combined_path)

        # Multi-sample comparison plots
        if len(results_dict) > 1:
            comparison_dir = output_dir / "comparisons"
            comparison_dir.mkdir(exist_ok=True)

            for plot_type in ['xanes', 'exafs', 'ft']:
                comp_path = comparison_dir / f"comparison_{plot_type}.png"
                self.plot_multi_sample_comparison(results_dict, plot_type,
                                                comp_path, show_plot=False)
                plot_files[f'comparison_{plot_type}'] = str(comp_path)

        return plot_files