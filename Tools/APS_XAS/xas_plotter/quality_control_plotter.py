"""
XAS Quality Control Plotting Module

Generates diagnostic plots for XAS data processing validation.
Shows raw data, deglitched data, normalized spectra with highlighted regions.

Optional execution - no plotting inside physics modules.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import yaml


class XASQualityControlPlotter:
    """
    Diagnostic plotting for XAS quality control.
    """

    def __init__(self,
                 plot_settings: Optional[Dict] = None,
                 save_path: Optional[str] = None):
        """
        Initialize quality control plotter.

        Parameters
        ----------
        plot_settings : dict, optional
            Plot styling settings
        save_path : str, optional
            Path to save plots
        """
        # Load YAML settings
        if plot_settings is None:
            settings_file = Path(__file__).parent.parent / "xas_config" / "xas_plot_settings.yaml"
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    yaml_settings = yaml.safe_load(f)
                self.yaml_settings = yaml_settings
                self.plot_settings = yaml_settings.get('plot_settings', {})
            except Exception as e:
                print(f"Warning: Could not load YAML settings: {e}")
                self.yaml_settings = {}
                self.plot_settings = {
                    "figsize": (14, 10),
                    "dpi": 150,
                    "fontsize": 16
                }
        else:
            self.plot_settings = plot_settings
            self.yaml_settings = {}
        
        self.save_path = Path(save_path) if save_path else None

    def plot_diagnostic_spectra(self,
                               energy: np.ndarray,
                               raw_mu: np.ndarray,
                               deglitched_mu: Optional[np.ndarray] = None,
                               normalized_mu: Optional[np.ndarray] = None,
                               normalization_params: Optional[Dict] = None,
                               sample_name: str = "XAS Spectrum",
                               save_plot: bool = True) -> plt.Figure:
        """
        Create diagnostic plot showing processing steps.

        Parameters
        ----------
        energy : np.ndarray
            Energy values
        raw_mu : np.ndarray
            Raw absorption data
        deglitched_mu : np.ndarray, optional
            Deglitched absorption
        normalized_mu : np.ndarray, optional
            Normalized absorption
        normalization_params : dict, optional
            Normalization parameters for highlighting regions
        sample_name : str
            Sample identifier
        save_plot : bool
            Whether to save the plot

        Returns
        -------
        fig : matplotlib Figure
            The diagnostic plot
        """
        # Get settings from YAML with correct fallback defaults
        figsize = tuple(self.plot_settings.get('figure_sizes', {}).get('combined_single', [14, 10]))
        dpi = self.plot_settings.get('dpi', 300)
        title_size = self.plot_settings.get('fonts', {}).get('title_size', 20)
        label_size = self.plot_settings.get('fonts', {}).get('axis_label_size', 18)
        tick_label_size = self.plot_settings.get('fonts', {}).get('tick_label_size', 16)
        line_width = self.plot_settings.get('lines', {}).get('data_line_width', 2)
        grid_alpha = self.plot_settings.get('lines', {}).get('grid_alpha', 0.3)
        colors = self.plot_settings.get('colors', {})
        
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        fig.suptitle(f"XAS Quality Control: {sample_name}", fontsize=title_size)

        # Plot 1: Raw μ(E)
        axes[0, 0].plot(energy, raw_mu, color=colors.get('data_line', 'blue'), 
                       linewidth=line_width, label='Raw μ(E)')
        axes[0, 0].set_title('Raw Absorption', fontsize=label_size)
        axes[0, 0].set_xlabel('Energy (eV)', fontsize=label_size)
        axes[0, 0].set_ylabel('μ(E)', fontsize=label_size)
        axes[0, 0].tick_params(axis='both', labelsize=tick_label_size)
        axes[0, 0].grid(True, alpha=grid_alpha)

        # Plot 2: Deglitched μ(E) if available
        if deglitched_mu is not None:
            axes[0, 1].plot(energy, raw_mu, color=colors.get('reference_line', 'darkblue'), 
                           linestyle='--', alpha=0.5, linewidth=line_width, label='Raw')
            axes[0, 1].plot(energy, deglitched_mu, color=colors.get('data_line', 'red'), 
                           linewidth=line_width, label='Deglitched')
            axes[0, 1].set_title('Deglitched Absorption', fontsize=label_size)
            axes[0, 1].set_xlabel('Energy (eV)', fontsize=label_size)
            axes[0, 1].set_ylabel('μ(E)', fontsize=label_size)
            axes[0, 1].tick_params(axis='both', labelsize=tick_label_size)
            axes[0, 1].legend(fontsize=self.plot_settings.get('legend', {}).get('fontsize', 16))
            axes[0, 1].grid(True, alpha=grid_alpha)
        else:
            axes[0, 1].plot(energy, raw_mu, color=colors.get('data_line', 'blue'), linewidth=line_width)
            axes[0, 1].set_title('Raw Absorption (No Deglitching)', fontsize=label_size)
            axes[0, 1].set_xlabel('Energy (eV)', fontsize=label_size)
            axes[0, 1].set_ylabel('μ(E)', fontsize=label_size)
            axes[0, 1].tick_params(axis='both', labelsize=tick_label_size)
            axes[0, 1].grid(True, alpha=grid_alpha)

        # Plot 3: Normalized XANES if available
        if normalized_mu is not None and normalization_params is not None:
            axes[1, 0].plot(energy, normalized_mu, color=colors.get('fit_line', 'green'), 
                           linewidth=line_width, label='Normalized')
            axes[1, 0].set_title('Normalized XANES', fontsize=label_size)
            axes[1, 0].set_xlabel('Energy (eV)', fontsize=label_size)
            axes[1, 0].set_ylabel('Normalized μ(E)', fontsize=label_size)
            axes[1, 0].tick_params(axis='both', labelsize=tick_label_size)
            axes[1, 0].grid(True, alpha=grid_alpha)

            # Highlight pre-edge and post-edge regions
            e0 = normalization_params.get("e0", np.median(energy))
            pre1 = normalization_params.get("pre1", -150)
            pre2 = normalization_params.get("pre2", -30)
            norm1 = normalization_params.get("norm1", 150)
            norm2 = normalization_params.get("norm2", 800)

            # Pre-edge region
            pre_mask = (energy >= e0 + pre1) & (energy <= e0 + pre2)
            if np.any(pre_mask):
                axes[1, 0].axvspan(energy[pre_mask][0], energy[pre_mask][-1],
                                  alpha=0.2, color=colors.get('data_line', 'blue'), label='Pre-edge')

            # Post-edge region
            post_mask = (energy >= e0 + norm1) & (energy <= e0 + norm2)
            if np.any(post_mask):
                axes[1, 0].axvspan(energy[post_mask][0], energy[post_mask][-1],
                                  alpha=0.2, color=colors.get('reference_line', 'red'), label='Post-edge')

            axes[1, 0].legend(fontsize=self.plot_settings.get('legend', {}).get('fontsize', 16))
        else:
            axes[1, 0].text(0.5, 0.5, 'No Normalization\nAvailable',
                           transform=axes[1, 0].transAxes, ha='center', va='center',
                           fontsize=label_size)
            axes[1, 0].set_title('Normalized XANES', fontsize=label_size)
            axes[1, 0].set_xlabel('Energy (eV)', fontsize=label_size)
            axes[1, 0].set_ylabel('Normalized μ(E)', fontsize=label_size)
            axes[1, 0].tick_params(axis='both', labelsize=tick_label_size)

        # Plot 4: Derivative analysis
        if normalized_mu is not None:
            dmu = np.gradient(normalized_mu, energy)
            axes[1, 1].plot(energy, dmu, color='purple', linewidth=line_width, label='dμ/dE')
            axes[1, 1].set_title('Derivative Analysis', fontsize=label_size)
            axes[1, 1].set_xlabel('Energy (eV)', fontsize=label_size)
            axes[1, 1].set_ylabel('dμ/dE', fontsize=label_size)
            axes[1, 1].tick_params(axis='both', labelsize=tick_label_size)
            axes[1, 1].grid(True, alpha=grid_alpha)

            # Mark E₀ if available
            if normalization_params and "e0" in normalization_params:
                e0 = normalization_params["e0"]
                axes[1, 1].axvline(e0, color=colors.get('reference_line', 'red'), 
                                  linestyle='--', alpha=0.7, label=f'E₀ = {e0:.1f} eV')
                axes[1, 1].legend(fontsize=self.plot_settings.get('legend', {}).get('fontsize', 16))
        else:
            dmu = np.gradient(raw_mu, energy)
            axes[1, 1].plot(energy, dmu, color='purple', linewidth=line_width, label='dμ/dE')
            axes[1, 1].set_title('Derivative Analysis (Raw)', fontsize=label_size)
            axes[1, 1].set_xlabel('Energy (eV)', fontsize=label_size)
            axes[1, 1].set_ylabel('dμ/dE', fontsize=label_size)
            axes[1, 1].tick_params(axis='both', labelsize=tick_label_size)
            axes[1, 1].grid(True, alpha=grid_alpha)

        plt.tight_layout()

        # Save plot if requested
        if save_plot and self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            plot_file = self.save_path / f"{sample_name}_diagnostic.png"
            export_settings = self.yaml_settings.get('export', {})
            fig.savefig(plot_file, 
                       dpi=dpi,
                       bbox_inches=export_settings.get('bbox_inches', 'tight'),
                       facecolor=export_settings.get('facecolor', 'white'),
                       transparent=export_settings.get('transparent', False))
            print(f"Diagnostic plot saved to {plot_file}")

        return fig


def create_xas_diagnostic_plot(energy: np.ndarray,
                              raw_mu: np.ndarray,
                              deglitched_mu: Optional[np.ndarray] = None,
                              normalized_mu: Optional[np.ndarray] = None,
                              normalization_params: Optional[Dict] = None,
                              sample_name: str = "XAS Spectrum",
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Convenience function for creating diagnostic plots.

    Parameters
    ----------
    energy : np.ndarray
        Energy values
    raw_mu : np.ndarray
        Raw absorption data
    deglitched_mu : np.ndarray, optional
        Deglitched absorption
    normalized_mu : np.ndarray, optional
        Normalized absorption
    normalization_params : dict, optional
        Normalization parameters
    sample_name : str
        Sample identifier
    save_path : str, optional
        Path to save plot

    Returns
    -------
    fig : matplotlib Figure
        The diagnostic plot
    """
    plotter = XASQualityControlPlotter(save_path=save_path)
    return plotter.plot_diagnostic_spectra(
        energy, raw_mu, deglitched_mu, normalized_mu,
        normalization_params, sample_name
    )