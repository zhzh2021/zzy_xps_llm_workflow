"""
XAS Data Analyzer Module

Core XAS data processing functionality following canonical workflow:
1. Energy alignment
2. Deglitching
3. Normalization
4. Validation
5. Quality assessment

Emits structured results for agent consumption.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd

# Import canonical workflow modules
try:
    from .energy_alignment import align_xas_energy
    from .deglitching import deglitch_xas_spectrum
    from .xas_normalization import normalize_xas_spectrum
    from .xas_normalization_validator import validate_normalization
    from .spectrum_quality_check import check_xas_spectrum_quality, XASQualityReportGenerator
except ImportError:
    from energy_alignment import align_xas_energy
    from deglitching import deglitch_xas_spectrum
    from xas_normalization import normalize_xas_spectrum
    from xas_normalization_validator import validate_normalization
    from spectrum_quality_check import check_xas_spectrum_quality, XASQualityReportGenerator


class XASProcessor:
    """
    XAS data processor following canonical workflow order.

    Executes: energy alignment → deglitching → normalization → validation → quality assessment
    Emits structured results for agent consumption.
    """

    def __init__(self,
                 alignment_method: str = 'derivative',
                 reference_energy: Optional[np.ndarray] = None,
                 reference_mu: Optional[np.ndarray] = None,
                 energy_shift: float = 0.0,
                 quality_thresholds: Optional[Dict] = None):
        """
        Initialize XAS processor.

        Parameters
        ----------
        alignment_method : str
            Method for energy alignment ('derivative' or 'reference')
        reference_energy : np.ndarray, optional
            Reference spectrum energy for alignment
        reference_mu : np.ndarray, optional
            Reference spectrum μ(E) for alignment
        """
        self.alignment_method = alignment_method
        self.reference_energy = reference_energy
        self.reference_mu = reference_mu
        self.energy_shift = energy_shift
        self.quality_thresholds = quality_thresholds

    def process_single_spectrum(self,
                               energy: np.ndarray,
                               mu: np.ndarray,
                               sample_name: str = "unknown") -> Dict[str, Any]:
        """
        Process a single XAS spectrum through canonical workflow.

        Executes: energy alignment → deglitching → normalization → validation → quality assessment

        Parameters
        ----------
        energy : np.ndarray
            Energy values in eV
        mu : np.ndarray
            Absorption coefficient μ(E)
        sample_name : str
            Name identifier for the sample

        Returns
        -------
        results : dict
            Structured processing results with confidence scores and flags
        """
        # Initialize result structure
        results = {
            "sample_name": sample_name,
            "energy_shift": {
                "applied": False,
                "shift_eV": 0.0
            },
            "energy_alignment": {},
            "deglitching": {},
            "normalization": {},
            "normalization_validation": {},
            "spectrum_quality": {},
            "success": False,
            "error": None
        }

        try:
            # Step 1: Energy Alignment
            if self.energy_shift and self.energy_shift != 0.0:
                energy = energy + self.energy_shift
                results["energy_shift"]["applied"] = True
                results["energy_shift"]["shift_eV"] = float(self.energy_shift)

            alignment_result = align_xas_energy(
                energy, mu, self.alignment_method,
                self.reference_energy, self.reference_mu
            )
            results["energy_alignment"] = alignment_result

            # Apply energy shift if needed
            if alignment_result["delta_e"] != 0:
                energy = energy + alignment_result["delta_e"]

            # Step 2: Deglitching
            deglitch_result = deglitch_xas_spectrum(energy, mu)
            results["deglitching"] = deglitch_result
            mu_cleaned = deglitch_result["cleaned_mu"]

            # Step 3: Normalization
            normalization_result = normalize_xas_spectrum(energy, mu_cleaned)
            results["normalization"] = normalization_result
            mu_normalized = normalization_result["normalized_mu"]

            # Step 4: Normalization Validation
            # Create Larch group for validation
            from larch import Group
            g = Group(energy=energy, mu=mu_cleaned)
            for k, v in normalization_result["parameters"].items():
                setattr(g, k, v)
            g.norm = mu_normalized

            validation_result = validate_normalization(g)
            results["normalization_validation"] = validation_result

            # Step 5: Spectrum Quality Check
            quality_result = check_xas_spectrum_quality(
                energy, mu_cleaned, mu_normalized,
                quality_thresholds=self.quality_thresholds,
                sample_id=sample_name, file_path=""
            )
            results["spectrum_quality"] = quality_result.to_dict()  # Convert to dict for JSON

            # Overall success
            results["success"] = True

            # Include processed data for downstream use
            results["processed_data"] = {
                "energy": energy,
                "mu_cleaned": mu_cleaned,
                "mu_normalized": mu_normalized
            }

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)

        return results

    def save_processed_data(self,
                           results: Dict[str, Any],
                           output_dir: str | Path,
                           formats: List[str] = None) -> Dict[str, str]:
        """
        Save processed XAS data to files.

        Parameters
        ----------
        results : dict
            Results from process_single_spectrum
        output_dir : str or Path
            Output directory
        formats : list of str
            File formats to save ('csv', 'txt', 'json')

        Returns
        -------
        saved_files : dict
            Dictionary mapping data types to saved file paths
        """
        if formats is None:
            formats = ['csv']

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        sample_name = results['sample_name']
        data = results.get('processed_data', {})

        saved_files = {}

        # Save normalized XANES
        if 'mu_normalized' in data and data['mu_normalized'] is not None:
            xanes_data = np.column_stack([data['energy'], data['mu_normalized']])
            for fmt in formats:
                if fmt == 'csv':
                    filename = f"{sample_name}_xanes.csv"
                    filepath = output_dir / filename
                    np.savetxt(filepath, xanes_data,
                              delimiter=",", header="energy_eV,mu_norm", comments="")
                    saved_files['xanes'] = str(filepath)

        return saved_files


    def create_feature_comparison_plots(self,
                                       results_dict: Dict[str, Dict],
                                       output_dir: str | Path = "feature_comparison_plots") -> Dict[str, str]:
        """
        Create comparison plots for XAS spectral features across multiple samples.

        Parameters
        ----------
        results_dict : dict
            Dictionary of processed results from process_single_spectrum
        output_dir : str or Path
            Output directory for plots

        Returns
        -------
        plot_files : dict
            Dictionary mapping plot types to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Extract features from all samples
        sample_features = {}
        for sample_name, results in results_dict.items():
            if not results.get('success', False):
                continue

            quality_data = results.get('spectrum_quality', {})
            if quality_data:
                sample_features[sample_name] = quality_data

        if not sample_features:
            print("No feature data available for comparison plots")
            return {}

        plot_files = {}

        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            import seaborn as sns

            # Set style
            plt.style.use('default')
            sns.set_palette("husl")

            # Feature categories to plot
            feature_groups = {
                'edge_features': {
                    'edge_position': 'Edge Position E₀ (eV)',
                    'edge_jump': 'Edge Jump (Δμ)',
                    'white_line_intensity': 'White Line Intensity'
                },
                'noise_features': {
                    'noise_level': 'Noise Level',
                    'signal_to_noise': 'Signal-to-Noise Ratio'
                },
                'slope_features': {
                    'pre_edge_slope': 'Pre-edge Slope',
                    'post_edge_slope': 'Post-edge Slope'
                },
                'quality_features': {
                    'normalization_quality': 'Normalization Quality',
                    'confidence': 'Quality Confidence'
                }
            }

            # Create comparison plots for each feature group
            for group_name, features in feature_groups.items():
                self._create_feature_group_plot(
                    sample_features, features, group_name,
                    output_dir, plot_files
                )

            # Create correlation heatmap
            self._create_feature_correlation_plot(
                sample_features, output_dir, plot_files
            )

            # Create radar/spider plot for multi-feature comparison
            self._create_radar_comparison_plot(
                sample_features, output_dir, plot_files
            )

        except ImportError as e:
            print(f"Warning: Missing plotting libraries: {e}")
        except Exception as e:
            print(f"Warning: Could not create feature comparison plots: {e}")

        return plot_files

    def _create_feature_group_plot(self,
                                  sample_features: Dict,
                                  features: Dict,
                                  group_name: str,
                                  output_dir: Path,
                                  plot_files: Dict):
        """Create bar plot for a group of related features."""
        try:
            import matplotlib.pyplot as plt

            n_features = len(features)
            n_samples = len(sample_features)

            if n_samples < 2:
                return

            fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
            if n_features == 1:
                axes = [axes]

            sample_names = list(sample_features.keys())

            for i, (feature_key, feature_label) in enumerate(features.items()):
                ax = axes[i]

                values = []
                valid_samples = []

                for sample in sample_names:
                    value = sample_features[sample].get(feature_key)
                    if value is not None and not np.isnan(value):
                        values.append(value)
                        valid_samples.append(sample)
                    else:
                        values.append(0)  # Placeholder for missing data
                        valid_samples.append(sample)

                if valid_samples:
                    bars = ax.bar(range(len(valid_samples)), values,
                                color=plt.cm.viridis(np.linspace(0, 1, len(valid_samples))),
                                alpha=0.7)
                    ax.set_xticks(range(len(valid_samples)))
                    ax.set_xticklabels(valid_samples, rotation=45, ha='right')
                    ax.set_ylabel(feature_label)
                    ax.set_title(f'{feature_label} Comparison')
                    ax.grid(True, alpha=0.3)

                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        if value != 0:  # Only label non-placeholder values
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                  '.3f', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            plot_file = output_dir / f"{group_name}_comparison.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plot_files[group_name] = str(plot_file)
            plt.close()

        except Exception as e:
            print(f"Warning: Could not create {group_name} plot: {e}")

    def _create_feature_correlation_plot(self,
                                        sample_features: Dict,
                                        output_dir: Path,
                                        plot_files: Dict):
        """Create correlation heatmap of all features."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd

            # Extract all numeric features
            feature_keys = ['edge_position', 'edge_jump', 'white_line_intensity',
                          'noise_level', 'signal_to_noise', 'pre_edge_slope',
                          'post_edge_slope', 'normalization_quality', 'confidence']

            # Create feature matrix
            feature_data = {}
            for sample, features in sample_features.items():
                for key in feature_keys:
                    value = features.get(key)
                    if value is not None and not np.isnan(value):
                        if key not in feature_data:
                            feature_data[key] = {}
                        feature_data[key][sample] = value

            if not feature_data:
                return

            # Create DataFrame
            df = pd.DataFrame(feature_data).T

            # Calculate correlation matrix
            corr_matrix = df.corr()

            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()

            plot_file = output_dir / "feature_correlations.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plot_files['correlations'] = str(plot_file)
            plt.close()

        except Exception as e:
            print(f"Warning: Could not create correlation plot: {e}")

    def _create_radar_comparison_plot(self,
                                     sample_features: Dict,
                                     output_dir: Path,
                                     plot_files: Dict):
        """Create radar plot for multi-feature comparison."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Select key features for radar plot
            radar_features = ['edge_jump', 'signal_to_noise', 'white_line_intensity',
                            'normalization_quality', 'confidence']

            # Prepare data
            feature_labels = ['Edge Jump', 'S/N Ratio', 'White Line', 'Norm Quality', 'Confidence']
            sample_names = list(sample_features.keys())

            if len(sample_names) < 2:
                return

            # Normalize features to 0-1 scale for radar plot
            normalized_data = {}
            for feature in radar_features:
                values = []
                for sample in sample_names:
                    value = sample_features[sample].get(feature, 0)
                    values.append(value)

                # Normalize to 0-1 range
                if values:
                    min_val, max_val = min(values), max(values)
                    if max_val > min_val:
                        normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
                    else:
                        normalized_values = [0.5] * len(values)  # All same value
                    normalized_data[feature] = normalized_values

            # Create radar plot
            angles = np.linspace(0, 2*np.pi, len(radar_features), endpoint=False).tolist()
            angles += angles[:1]  # Close the plot

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

            colors = plt.cm.viridis(np.linspace(0, 1, len(sample_names)))

            for i, sample in enumerate(sample_names):
                values = [normalized_data[feature][i] for feature in radar_features]
                values += values[:1]  # Close the plot

                ax.plot(angles, values, 'o-', linewidth=2, label=sample, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(feature_labels)
            ax.set_ylim(0, 1)
            ax.set_title('Multi-Feature Comparison', size=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax.grid(True)

            plot_file = output_dir / "radar_comparison.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plot_files['radar'] = str(plot_file)
            plt.close()

        except Exception as e:
            print(f"Warning: Could not create radar plot: {e}")


def create_summary_table(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a summary table from batch processing results.

    Parameters
    ----------
    results_dict : dict
        Results from process_batch

    Returns
    -------
    summary_df : pd.DataFrame
        Summary table with extracted features
    """
    features_list = []

    for sample_name, results in results_dict.items():
        # Extract features from spectrum_quality data
        quality_data = results.get('spectrum_quality', {})

        # Add sample name and flatten quality metrics
        features = {'sample': sample_name}
        features.update(quality_data)

        features_list.append(features)

    if not features_list:
        return pd.DataFrame()

    summary_df = pd.DataFrame(features_list)
    return summary_df


def create_feature_comparison_plots(results_dict: Dict[str, Dict],
                                   output_dir: str | Path = "feature_comparison_plots") -> Dict[str, str]:
    """
    Convenience function to create feature comparison plots across multiple XAS samples.

    Generates comprehensive comparison plots for spectral features including:
    - Edge features (E₀, edge jump, white line intensity)
    - Noise characteristics (noise level, signal-to-noise ratio)
    - Slope features (pre/post edge slopes)
    - Quality metrics (normalization quality, confidence)
    - Feature correlations heatmap
    - Radar plot for multi-feature comparison

    Parameters
    ----------
    results_dict : dict
        Dictionary of processed XAS results from XASProcessor.process_single_spectrum
    output_dir : str or Path
        Output directory for plots (default: "feature_comparison_plots")

    Returns
    -------
    plot_files : dict
        Dictionary mapping plot types to file paths

    Example
    -------
    >>> from xas_analyzer_main import create_feature_comparison_plots
    >>> results = processor.process_batch(sample_files)
    >>> plots = create_feature_comparison_plots(results)
    >>> print(f"Created {len(plots)} comparison plots")
    """
    processor = XASProcessor()
    return processor.create_feature_comparison_plots(results_dict, output_dir)