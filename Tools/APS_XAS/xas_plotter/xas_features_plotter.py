"""this module is to plot extracted XAS features such as E0 on XAS spectra that xas_plotter_main can call."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import matplotlib.gridspec as gridspec
import yaml
import pandas as pd

# Handle optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using default matplotlib styling")

# Try to import XASBatchResults, but make it optional
try:
    from ..xas_analyzer.xas_batch_results import XASBatchResults
    HAS_BATCH_RESULTS = True
except ImportError:
    try:
        from ...xas_analyzer.xas_batch_results import XASBatchResults
        HAS_BATCH_RESULTS = True
    except ImportError:
        HAS_BATCH_RESULTS = False
        XASBatchResults = None   

def create_feature_comparison_plots(batch_results: Union[Dict[str, Any], Any],
                                   output_dir: str | Path = "feature_comparison_plots",
                                   export_csv: bool = True) -> Dict[str, str]:
    """
    Create comprehensive comparison plots for XAS features across multiple samples.
    Optionally exports features to CSV using the data models.

    Generates plots for:
    - Edge features (E₀, edge step, white line)
    - XANES features (area, white line intensity/energy)
    - EXAFS features (χ(k) RMS, FT peaks)
    - Feature correlations
    - Multi-feature radar plots

    Parameters
    ----------
    batch_results : dict or XASBatchResults
        Batch results containing features for multiple samples.
        Can be either a dictionary with 'samples' key or XASBatchResults object.
    output_dir : str or Path
        Output directory for plots and CSV files
    export_csv : bool
        Whether to export features to CSV file (default True)

    Returns
    -------
    plot_files : dict
        Dictionary mapping plot types to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Extract features from all samples - handle both dict and object formats
    sample_features = {}
    
    # Determine input format and extract samples
    if hasattr(batch_results, 'samples'):  # XASBatchResults object
        samples_data = batch_results.samples
    elif isinstance(batch_results, dict) and 'samples' in batch_results:
        samples_data = batch_results['samples']
    elif isinstance(batch_results, dict):
        # Assume the dict itself contains sample data
        samples_data = batch_results
    else:
        print("Error: Unsupported batch results format")
        return {}
        
    for sample_name, sample_result in samples_data.items():
        # Handle both object and dictionary feature access
        if hasattr(sample_result, 'features'):
            features = sample_result.features
            sample_features[sample_name] = {
                'e0': getattr(features, 'e0', None),
                'edge_step': getattr(features, 'edge_step', None),
                'white_line_intensity': getattr(features, 'white_line_intensity', None),
                'white_line_energy': getattr(features, 'white_line_energy', None),
                'xanes_area': getattr(features, 'xanes_area', None),
                'chi_k_rms': getattr(features, 'chi_k_rms', None),
                'ft_peak_r': getattr(features, 'ft_peak_r', None),
                'ft_peak_amp': getattr(features, 'ft_peak_amp', None),
                'ft_area': getattr(features, 'ft_area', None)
            }
        elif isinstance(sample_result, dict) and 'features' in sample_result:
            features = sample_result['features']
            sample_features[sample_name] = {
                'e0': features.get('e0'),
                'edge_step': features.get('edge_step'),
                'white_line_intensity': features.get('white_line_intensity'),
                'white_line_energy': features.get('white_line_energy'),
                'xanes_area': features.get('xanes_area'),
                'chi_k_rms': features.get('chi_k_rms'),
                'ft_peak_r': features.get('ft_peak_r'),
                'ft_peak_amp': features.get('ft_peak_amp'),
                'ft_area': features.get('ft_area')
            }
        else:
            print(f"Warning: Could not extract features for sample {sample_name}")
            continue

    if not sample_features:
        print("No feature data available for comparison plots")
        return {}

    plot_files = {}

    # Export features to CSV if requested using the models module
    if export_csv:
        try:
            # Try to use the structured batch results for CSV export
            if hasattr(batch_results, 'to_csv'):  # XASBatchResults object
                csv_path = batch_results.to_csv(output_dir / "xas_features_comparison.csv")
                plot_files['features_csv'] = str(csv_path)
                print(f"Exported features to CSV: {csv_path}")
            else:
                # For dictionary inputs, create a simple CSV export
                try:
                    import pandas as pd
                    # Create simple DataFrame from sample_features
                    df = pd.DataFrame.from_dict(sample_features, orient='index')
                    df.reset_index(inplace=True)
                    df.rename(columns={'index': 'sample_name'}, inplace=True)
                    
                    csv_path = output_dir / "xas_features_comparison.csv"
                    df.to_csv(csv_path, index=False, float_format='%.6f')
                    plot_files['features_csv'] = str(csv_path)
                    print(f"Exported features to CSV: {csv_path}")
                except ImportError:
                    print("Warning: pandas not available for CSV export")
        except Exception as e:
            print(f"Warning: Could not export features to CSV: {e}")

    try:
        # Set plotting style
        plt.style.use('default')
        if HAS_SEABORN:
            sns.set_palette("husl")

        # Create different types of comparison plots
        plot_files.update(_create_edge_feature_plots(sample_features, output_dir))
        plot_files.update(_create_xanes_feature_plots(sample_features, output_dir))
        plot_files.update(_create_exafs_feature_plots(sample_features, output_dir))
        plot_files.update(_create_correlation_plots(sample_features, output_dir))
        plot_files.update(_create_radar_comparison_plot(sample_features, output_dir))

        print(f"Created {len([k for k in plot_files.keys() if k != 'features_csv'])} feature comparison plots in {output_dir}")

    except ImportError as e:
        print(f"Warning: Missing plotting libraries: {e}")
    except Exception as e:
        print(f"Warning: Could not create feature comparison plots: {e}")

    return plot_files



def _create_edge_feature_plots(sample_features: Dict, output_dir: Path) -> Dict[str, str]:
    """Create plots for edge-related features."""
    plot_files = {}

    try:
        edge_features = {
            'e0': 'Edge Energy E₀ (eV)',
            'edge_step': 'Edge Step Height',
            'white_line_intensity': 'White Line Intensity',
            'white_line_energy': 'White Line Energy (eV)'
        }

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        sample_names = list(sample_features.keys())

        for i, (feature_key, feature_label) in enumerate(edge_features.items()):
            ax = axes[i]

            values = []
            valid_samples = []

            for sample in sample_names:
                value = sample_features[sample].get(feature_key)
                if value is not None:
                    values.append(value)
                    valid_samples.append(sample)

            if valid_samples:
                bars = ax.bar(range(len(valid_samples)), values,
                            color=plt.cm.viridis(np.linspace(0, 1, len(valid_samples))),
                            alpha=0.7)
                ax.set_xticks(range(len(valid_samples)))
                ax.set_xticklabels(valid_samples, rotation=45, ha='right')
                ax.set_ylabel(feature_label)
                ax.set_title(f'{feature_label}')
                ax.grid(True, alpha=0.3)

                # Add value labels
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                          '.2f', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plot_file = output_dir / "edge_features_comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plot_files['edge_features'] = str(plot_file)
        plt.close()

    except Exception as e:
        print(f"Warning: Could not create edge feature plots: {e}")

    return plot_files


def _create_xanes_feature_plots(sample_features: Dict, output_dir: Path) -> Dict[str, str]:
    """Create plots for XANES-related features."""
    plot_files = {}

    try:
        xanes_features = {
            'xanes_area': 'XANES Integrated Area',
            'white_line_intensity': 'White Line Intensity',
            'white_line_energy': 'White Line Energy (eV)'
        }

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        sample_names = list(sample_features.keys())

        for i, (feature_key, feature_label) in enumerate(xanes_features.items()):
            ax = axes[i]

            values = []
            valid_samples = []

            for sample in sample_names:
                value = sample_features[sample].get(feature_key)
                if value is not None:
                    values.append(value)
                    valid_samples.append(sample)

            if valid_samples:
                bars = ax.bar(range(len(valid_samples)), values,
                            color=plt.cm.plasma(np.linspace(0, 1, len(valid_samples))),
                            alpha=0.7)
                ax.set_xticks(range(len(valid_samples)))
                ax.set_xticklabels(valid_samples, rotation=45, ha='right')
                ax.set_ylabel(feature_label)
                ax.set_title(f'{feature_label}')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = output_dir / "xanes_features_comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plot_files['xanes_features'] = str(plot_file)
        plt.close()

    except Exception as e:
        print(f"Warning: Could not create XANES feature plots: {e}")

    return plot_files


def _create_exafs_feature_plots(sample_features: Dict, output_dir: Path) -> Dict[str, str]:
    """Create plots for EXAFS-related features."""
    plot_files = {}

    try:
        exafs_features = {
            'chi_k_rms': 'χ(k) RMS Amplitude',
            'ft_peak_r': 'FT Peak Position R (Å)',
            'ft_peak_amp': 'FT Peak Amplitude',
            'ft_area': 'FT Integrated Area'
        }

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        sample_names = list(sample_features.keys())

        for i, (feature_key, feature_label) in enumerate(exafs_features.items()):
            ax = axes[i]

            values = []
            valid_samples = []

            for sample in sample_names:
                value = sample_features[sample].get(feature_key)
                if value is not None:
                    values.append(value)
                    valid_samples.append(sample)

            if valid_samples:
                bars = ax.bar(range(len(valid_samples)), values,
                            color=plt.cm.cool(np.linspace(0, 1, len(valid_samples))),
                            alpha=0.7)
                ax.set_xticks(range(len(valid_samples)))
                ax.set_xticklabels(valid_samples, rotation=45, ha='right')
                ax.set_ylabel(feature_label)
                ax.set_title(f'{feature_label}')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = output_dir / "exafs_features_comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plot_files['exafs_features'] = str(plot_file)
        plt.close()

    except Exception as e:
        print(f"Warning: Could not create EXAFS feature plots: {e}")

    return plot_files


def _create_correlation_plots(sample_features: Dict, output_dir: Path) -> Dict[str, str]:
    """Create correlation plots between features."""
    plot_files = {}

    try:
        import pandas as pd

        # Create feature DataFrame
        df = pd.DataFrame.from_dict(sample_features, orient='index')

        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols]

        if len(df_numeric.columns) > 1:
            # Correlation heatmap
            plt.figure(figsize=(10, 8))
            corr_matrix = df_numeric.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()

            plot_file = output_dir / "feature_correlations.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plot_files['correlations'] = str(plot_file)
            plt.close()

            # Pair plot for key features
            key_features = ['e0', 'edge_step', 'white_line_intensity', 'xanes_area']
            available_features = [f for f in key_features if f in df_numeric.columns and df_numeric[f].notna().sum() > 1]

            if len(available_features) > 1:
                plt.figure(figsize=(12, 10))
                pair_plot = sns.pairplot(df_numeric[available_features], diag_kind='kde')
                pair_plot.fig.suptitle('Feature Pairwise Relationships', y=1.02)
                plt.tight_layout()

                plot_file = output_dir / "feature_pairs.png"
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plot_files['pairs'] = str(plot_file)
                plt.close()

    except Exception as e:
        print(f"Warning: Could not create correlation plots: {e}")

    return plot_files


def _create_radar_comparison_plot(sample_features: Dict, output_dir: Path) -> Dict[str, str]:
    """Create radar plot for multi-feature comparison."""
    plot_files = {}

    try:
        # Select key features for radar plot
        radar_features = ['edge_step', 'white_line_intensity', 'xanes_area', 'chi_k_rms']

        # Prepare data
        feature_labels = ['Edge Step', 'White Line', 'XANES Area', 'χ(k) RMS']
        sample_names = list(sample_features.keys())

        if len(sample_names) < 2:
            return plot_files

        # Normalize features to 0-1 scale for radar plot
        normalized_data = {}
        for feature in radar_features:
            values = []
            for sample in sample_names:
                value = sample_features[sample].get(feature, 0)
                if value is not None:
                    values.append(value)
                else:
                    values.append(0)

            # Normalize to 0-1 range
            if values and max(values) > min(values):
                min_val, max_val = min(values), max(values)
                normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                normalized_values = [0.5] * len(values)  # All same value
            normalized_data[feature] = normalized_values

        # Create radar plot
        angles = np.linspace(0, 2*np.pi, len(radar_features), endpoint=False).tolist()
        angles += angles[:1]  # Close the plot

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

        colors = plt.cm.Set2(np.linspace(0, 1, len(sample_names)))

        for i, sample in enumerate(sample_names):
            values = [normalized_data[feature][i] for feature in radar_features]
            values += values[:1]  # Close the plot

            ax.plot(angles, values, 'o-', linewidth=2, label=sample, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Feature Comparison Across Samples', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plot_file = output_dir / "radar_comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plot_files['radar'] = str(plot_file)
        plt.close()

    except Exception as e:
        print(f"Warning: Could not create radar plot: {e}")

    return plot_files
