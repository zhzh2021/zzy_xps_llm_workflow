"""
XAS Analyzer - Main Entry Point

Reads standardized XAS datasets and performs:
1. Energy alignment
2. Deglitching  
3. Normalization
4. Edge step calculation
5. Quality assessment
6. Saves processed results

Usage:
    # From Python
    from main import analyze_dataset, batch_analyze
    
    # Single file
    results = analyze_dataset("JL10_1_standardized.csv")
    
    # Batch processing
    all_results = batch_analyze()
    
    # Command line
    python main.py --file "JL10_1_standardized.csv"
    python main.py  # Process all files
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import re
import pandas as pd
import numpy as np
import xarray as xr
import yaml

# Import the processor
from xas_analyzer_main import XASProcessor

# Optional data exclusion module for downstream analysis
try:
    from xas_exclude_measurement import (
        XASDataExcluder,
        exclude_bad_measurements_manual,
        exclude_bad_measurements_auto
    )
    HAS_EXCLUDER = True
except ImportError:
    XASDataExcluder = None
    exclude_bad_measurements_manual = None
    exclude_bad_measurements_auto = None
    HAS_EXCLUDER = False

# Optional averaging module for grouped measurements
try:
    from xas_average_datagroup import average_spectra
    HAS_AVERAGER = True
except ImportError:
    average_spectra = None
    HAS_AVERAGER = False

# Import plotting modules (optional)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'xas_plotter'))
    from quality_control_plotter import XASQualityControlPlotter
    HAS_PLOTTER = True
except ImportError:
    XASQualityControlPlotter = None
    HAS_PLOTTER = False
    print("Warning: Quality control plotter not available")

__version__ = '3.0'

# Global variable for timestamp folder (set once per batch run)
_TIMESTAMP_FOLDER = None


def get_timestamp_folder() -> str:
    """Get or create timestamp folder name for this analysis session."""
    global _TIMESTAMP_FOLDER
    if _TIMESTAMP_FOLDER is None:
        _TIMESTAMP_FOLDER = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _TIMESTAMP_FOLDER


def reset_timestamp():
    """Reset timestamp for new batch run."""
    global _TIMESTAMP_FOLDER
    _TIMESTAMP_FOLDER = None


def get_standardized_data_dir() -> Path:
    """Get the directory containing standardized XAS data."""
    current_file = Path(__file__).resolve()
    zzy_llm_dir = current_file.parent.parent.parent.parent  # xas_analyzer -> APS_XAS -> Tools -> zzy_llm
    return zzy_llm_dir / "project_root" / "xas_results" / "01_standardized_data"


def get_output_dir() -> Path:
    """Get the output directory for analyzed data."""
    current_file = Path(__file__).resolve()
    zzy_llm_dir = current_file.parent.parent.parent.parent
    return zzy_llm_dir / "project_root" / "xas_results" / "02_analyzed_data"

def load_pipeline_config(config_path: str | Path | None = None) -> dict:
    """Load analyzer config (energy shift, etc.)."""
    if config_path is None:
        current_file = Path(__file__).resolve()
        zzy_llm_dir = current_file.parent.parent.parent.parent
        config_path = zzy_llm_dir / 'Tools' / 'APS_XAS' / 'xas_config' / 'pipeline_config.yaml'
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    try:
        import yaml
        data = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_standardized_csv(file_path: Path) -> xr.Dataset:
    """
    Load standardized CSV file into xarray.Dataset.
    
    Parameters
    ----------
    file_path : Path
        Path to standardized CSV file
    
    Returns
    -------
    xr.Dataset
        XAS dataset with energy coordinate and data variables
    """
    # Read CSV
    df = pd.read_csv(file_path)
    
    # Extract energy
    energy = df['energy'].values
    
    # Create dataset
    data_vars = {}
    for col in df.columns:
        if col != 'energy':
            data_vars[col] = (['point'], df[col].values)
    
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={'energy': ('point', energy)},
        attrs={
            'filename': file_path.name,
            'source_file': str(file_path)
        }
    )
    
    return ds


def analyze_dataset(file_path: str | Path,
                   output_dir: Optional[str | Path] = None,
                   save: bool = True,
                   plot: bool = True,
                   config: Optional[dict] = None) -> Dict[str, Any]:
    """
    Analyze a single standardized XAS dataset.
    
    Loads CSV, runs XASProcessor pipeline (alignment → deglitching → 
    normalization → validation → quality check), and optionally generates
    diagnostic plots using xas_plot_settings.yaml configuration.
    
    Parameters
    ----------
    file_path : str or Path
        Path to standardized CSV file (relative to standardized data dir or absolute)
    output_dir : str or Path, optional
        Output directory. Default: project_root/xas_results/02_analyzed_data
    save : bool, default True
        Whether to save processed results (CSV + JSON)
    plot : bool, default False
        Whether to generate quality control diagnostic plots
        Requires: quality_control_plotter.py and xas_plot_settings.yaml
    
    Returns
    -------
    dict
        Analysis results including normalized spectrum and quality metrics
    
    Examples
    --------
    >>> # Basic analysis
    >>> results = analyze_dataset("JL10_1_standardized.csv")
    >>> print(results['normalization']['parameters']['edge_step'])
    
    >>> # With diagnostic plots
    >>> results = analyze_dataset("JL10_1_standardized.csv", plot=True)
    """
    # Resolve file path
    file_path = Path(file_path)
    if not file_path.is_absolute():
        file_path = get_standardized_data_dir() / file_path
    
    if not file_path.exists():
        raise FileNotFoundError(f"Standardized data file not found: {file_path}")
    
    # Load dataset
    ds = load_standardized_csv(file_path)
    
    # Extract energy and mu
    energy = ds.energy.values
    mu = ds.mu_trans.values
    
    # Create processor and run analysis
    energy_shift = 0.0
    quality_thresholds = {}
    if config:
        energy_shift = float(config.get('pipeline', {}).get('preprocessing', {}).get('energy_shift', 0.0) or 0.0)
        quality_thresholds = (
            config.get('pipeline', {}).get('quality_thresholds') or
            config.get('pipeline', {}).get('quality') or
            config.get('quality_thresholds') or
            config.get('quality') or
            config.get('feature_extraction', {}).get('quality') or
            {}
        )
    processor = XASProcessor(energy_shift=energy_shift, quality_thresholds=quality_thresholds)
    results = processor.process_single_spectrum(energy, mu, sample_name=file_path.stem)
    
    # Add source information
    results['source_file'] = file_path.name
    results['dataset_variables'] = list(ds.data_vars.keys())
    
    # Setup output directory
    if output_dir is None:
        output_dir = get_output_dir()
    else:
        output_dir = Path(output_dir)
    
    if save or plot:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate quality control plots if requested
    if plot and HAS_PLOTTER and results['success']:
        try:
            # Create plots subdirectory structure
            plots_dir = output_dir / "plots" / "single_spectrum"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            plotter = XASQualityControlPlotter(save_path=plots_dir)
            plot_name = file_path.stem.replace('_standardized', '')
            
            plotter.plot_diagnostic_spectra(
                energy=results['processed_data']['energy'],
                raw_mu=mu,
                deglitched_mu=results['processed_data']['mu_cleaned'],
                normalized_mu=results['processed_data']['mu_normalized'],
                normalization_params=results['normalization']['parameters'],
                sample_name=plot_name,
                save_plot=True
            )
            print(f"Saved diagnostic plot to: {plots_dir}")
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    # Save if requested
    if save:
        # Create organized subdirectories
        normalized_dir = output_dir / "normalized_data"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        
        quality_reports_dir = output_dir / "quality_reports" / get_timestamp_folder()
        quality_reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save normalized spectrum as CSV in normalized_data/
        output_name = file_path.stem.replace('_standardized', '_analyzed.csv')
        output_path = normalized_dir / output_name
        
        processed_df = pd.DataFrame({
            'energy': results['processed_data']['energy'],
            'mu_cleaned': results['processed_data']['mu_cleaned'],
            'mu_normalized': results['processed_data']['mu_normalized']
        })
        processed_df.to_csv(output_path, index=False)
        print(f"Saved analyzed data to: {output_path}")
        
        # Save quality report as JSON in quality_reports/<timestamp>/
        import json
import re
        report_path = quality_reports_dir / file_path.stem.replace('_standardized', '_quality_report.json')
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = results.copy()
        if 'processed_data' in results_json:
            del results_json['processed_data']  # Too large for JSON
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        results_json = convert_to_serializable(results_json)
        
        with open(report_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"Saved quality report to: {report_path}")
    
    # Print summary
    print(f"\nAnalysis complete for {results['sample_name']}:")
    print(f"  Success: {results['success']}")
    
    if results['success']:
        print(f"\n  Normalization:")
        norm_params = results['normalization']['parameters']
        print(f"    E0: {norm_params['e0']:.2f} eV")
        print(f"    Edge step: {norm_params['edge_step']:.4f}")
        print(f"    Confidence: {results['normalization']['confidence']:.2f}")
        
        print(f"\n  Quality:")
        quality = results['spectrum_quality']
        print(f"    Classification: {quality['classification']}")
        print(f"    Signal-to-noise: {quality['signal_to_noise']:.2f}")
        print(f"    Edge jump: {quality['edge_jump']:.4f}")
        print(f"    Suitable for analysis: {quality['suitable_for_analysis']}")
    
    return results


def batch_analyze(pattern: str = "*_standardized.csv",
                 output_dir: Optional[str | Path] = None,
                 save: bool = True,
                 plot: bool = True,
                 config: Optional[dict] = None) -> List[Dict[str, Any]]:
    """
    Analyze multiple standardized XAS datasets.
    
    Parameters
    ----------
    pattern : str, default "*_standardized.csv"
        Glob pattern for file selection
    output_dir : str or Path, optional
        Output directory. Default: project_root/xas_results/02_analyzed_data
    save : bool, default True
        Whether to save processed results
    plot : bool, default False
        Whether to generate quality control diagnostic plots for each sample
    
    Returns
    -------
    List[dict]
        List of analysis results for each dataset
    
    Examples
    --------
    >>> results = batch_analyze()
    >>> print(f"Processed {len(results)} datasets")
    
    >>> # With plotting
    >>> results = batch_analyze(plot=True)
    """
    data_dir = get_standardized_data_dir()
    files = sorted(data_dir.glob(pattern))
    
    # Setup output directory
    if output_dir is None:
        output_dir = get_output_dir()
    else:
        output_dir = Path(output_dir)
    
    # Reset timestamp for new batch run
    reset_timestamp()
    timestamp = get_timestamp_folder()
    print(f"Batch analysis session: {timestamp}")
    
    all_results = []
    for file_path in files:
        try:
            print(f"\nProcessing: {file_path.name}")
            results = analyze_dataset(file_path, output_dir=output_dir, save=save, plot=plot, config=config)
            all_results.append(results)
            
        except Exception as e:
            print(f"  [Failed] Error: {e}")
    
    print(f"\nSuccessfully analyzed {len(all_results)} out of {len(files)} datasets")
    
    # Save batch summary reports if there are results
    if save and len(all_results) > 0:
        try:
            _save_batch_summary_reports(all_results, output_dir)
        except Exception as e:
            print(f"\nWarning: Could not save batch summary reports: {e}")
    
    # Generate batch summary plots if plotting is enabled
    if plot and len(all_results) > 1:
        try:
            _create_batch_summary_plots(all_results, output_dir)
        except Exception as e:
            print(f"\nWarning: Could not create batch summary plots: {e}")
    
    return all_results


def _save_batch_summary_reports(all_results: List[Dict[str, Any]], output_dir: Path):
    """
    Save batch summary reports (CSV and TXT) in timestamped quality_reports folder.
    
    Parameters
    ----------
    all_results : List[Dict]
        Results from batch analysis
    output_dir : Path
        Base output directory
    """
    # Get timestamped quality reports directory
    quality_reports_dir = output_dir / "quality_reports" / get_timestamp_folder()
    quality_reports_dir.mkdir(parents=True, exist_ok=True)
    
    successful_results = [r for r in all_results if r.get('success', False)]
    if not successful_results:
        return
    
    # Create CSV summary
    csv_data = []
    for res in successful_results:
        row = {
            'sample_name': res['sample_name'].replace('_standardized', ''),
            'e0_eV': res['normalization']['parameters']['e0'],
            'edge_step': res['normalization']['parameters']['edge_step'],
            'norm_confidence': res['normalization']['confidence'],
            'classification': res['spectrum_quality']['classification'],
            'signal_to_noise': res['spectrum_quality']['signal_to_noise'],
            'edge_jump': res['spectrum_quality']['edge_jump'],
            'noise_level': res['spectrum_quality']['noise_level'],
            'data_points': res['spectrum_quality']['data_points'],
            'suitable_for_analysis': res['spectrum_quality']['suitable_for_analysis']
        }
        csv_data.append(row)
    
    import pandas as pd
    df = pd.DataFrame(csv_data)
    
    # Save CSV
    csv_path = quality_reports_dir / "xas_batch_quality_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved batch quality CSV to: {csv_path}")
    
    # Save text summary
    txt_path = quality_reports_dir / "xas_batch_quality_report_summary.txt"
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"XAS BATCH ANALYSIS SUMMARY - {get_timestamp_folder()}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total samples processed: {len(all_results)}\n")
        f.write(f"Successful analyses: {len(successful_results)}\n")
        f.write(f"Failed analyses: {len(all_results) - len(successful_results)}\n\n")
        
        f.write("Quality Classification Distribution:\n")
        classifications = [r['spectrum_quality']['classification'] for r in successful_results]
        from collections import Counter
        for classification, count in Counter(classifications).most_common():
            f.write(f"  {classification}: {count}\n")
        
        f.write(f"\nE0 Range: {df['e0_eV'].min():.2f} - {df['e0_eV'].max():.2f} eV\n")
        f.write(f"Average SNR: {df['signal_to_noise'].mean():.2f}\n")
        f.write(f"Average Edge Step: {df['edge_step'].mean():.4f}\n\n")
        
        f.write("Samples suitable for analysis: ")
        f.write(f"{df['suitable_for_analysis'].sum()} / {len(df)}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Saved batch summary text to: {txt_path}")


def _create_batch_summary_plots(all_results: List[Dict[str, Any]], output_dir: Path):
    """
    Create batch summary plots: group comparison and quality overview.
    
    Parameters
    ----------
    all_results : List[Dict]
        Results from batch analysis
    output_dir : Path
        Base output directory
    """
    # Filter successful results
    successful_results = [r for r in all_results if r.get('success', False)]
    if not successful_results:
        print("No successful results to plot")
        return
    
    # Create group comparison plot (all normalized spectra overlaid)
    try:
        group_dir = output_dir / "plots" / "group_comparison"
        group_dir.mkdir(parents=True, exist_ok=True)
        _plot_group_comparison(successful_results, group_dir)
        print(f"\nSaved group comparison plot to: {group_dir}")
    except Exception as e:
        print(f"Warning: Could not create group comparison plot: {e}")
    
    # Create quality overview plots
    try:
        quality_dir = output_dir / "plots" / "quality_plots"
        quality_dir.mkdir(parents=True, exist_ok=True)
        _plot_quality_overview(successful_results, quality_dir)
        print(f"Saved quality overview plots to: {quality_dir}")
    except Exception as e:
        print(f"Warning: Could not create quality overview plots: {e}")


def _plot_group_comparison(results: List[Dict[str, Any]], output_dir: Path):
    """Plot all normalized spectra overlaid in one figure."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    
    for i, res in enumerate(results):
        energy = res['processed_data']['energy']
        mu_norm = res['processed_data']['mu_normalized']
        label = res['sample_name'].replace('_standardized', '')
        
        ax.plot(energy, mu_norm, linewidth=2, alpha=0.7, 
                color=colors[i], label=label)
    
    ax.set_xlabel('Energy (eV)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized μ(E)', fontsize=14, fontweight='bold')
    ax.set_title(f'Group Comparison - All Normalized Spectra (N={len(results)})',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_normalized_spectra.png', dpi=300, bbox_inches='tight')
    plt.close()


def _plot_quality_overview(results: List[Dict[str, Any]], output_dir: Path):
    """Create quality metric overview plots."""
    import matplotlib.pyplot as plt
    
    # Extract quality metrics
    sample_names = [r['sample_name'].replace('_standardized', '') for r in results]
    e0_values = [r['normalization']['parameters']['e0'] for r in results]
    edge_steps = [r['normalization']['parameters']['edge_step'] for r in results]
    snr_values = [r['spectrum_quality']['signal_to_noise'] for r in results]
    edge_jumps = [r['spectrum_quality']['edge_jump'] for r in results]
    classifications = [r['spectrum_quality']['classification'] for r in results]
    
    # Create 2x2 subplot for quality metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Batch Quality Assessment Overview', fontsize=18, fontweight='bold')
    
    # Plot 1: E0 distribution
    ax1 = axes[0, 0]
    ax1.bar(range(len(sample_names)), e0_values, color='steelblue', alpha=0.7)
    ax1.set_ylabel('E₀ (eV)', fontsize=12, fontweight='bold')
    ax1.set_title('Edge Position (E₀)', fontsize=14)
    ax1.set_xticks(range(len(sample_names)))
    ax1.set_xticklabels(sample_names, rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Edge step distribution
    ax2 = axes[0, 1]
    ax2.bar(range(len(sample_names)), edge_steps, color='coral', alpha=0.7)
    ax2.set_ylabel('Edge Step', fontsize=12, fontweight='bold')
    ax2.set_title('Edge Step Values', fontsize=14)
    ax2.set_xticks(range(len(sample_names)))
    ax2.set_xticklabels(sample_names, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: SNR vs Edge Jump scatter
    ax3 = axes[1, 0]
    scatter = ax3.scatter(snr_values, edge_jumps, c=range(len(results)), 
                         cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    for i, name in enumerate(sample_names):
        ax3.annotate(name, (snr_values[i], edge_jumps[i]), 
                    fontsize=7, alpha=0.7, xytext=(5, 5), 
                    textcoords='offset points')
    ax3.set_xlabel('Signal-to-Noise Ratio', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Edge Jump', fontsize=12, fontweight='bold')
    ax3.set_title('SNR vs Edge Jump', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Classification summary
    ax4 = axes[1, 1]
    from collections import Counter
    class_counts = Counter(classifications)
    colors_pie = {'excellent': 'green', 'good': 'lightgreen', 
                  'acceptable': 'yellow', 'poor': 'orange', 'invalid': 'red'}
    pie_colors = [colors_pie.get(c, 'gray') for c in class_counts.keys()]
    ax4.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%',
           colors=pie_colors, startangle=90)
    ax4.set_title('Quality Classification Distribution', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create trends plot (E0 and edge step trends)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Quality Metrics Trends', fontsize=18, fontweight='bold')
    
    x_pos = range(len(sample_names))
    
    # Trend 1: E0 with error bars (using edge step as proxy for variation)
    ax1 = axes[0]
    ax1.plot(x_pos, e0_values, 'o-', linewidth=2, markersize=8, color='navy', alpha=0.7)
    ax1.fill_between(x_pos, 
                     [e - 0.5 for e in e0_values], 
                     [e + 0.5 for e in e0_values], 
                     alpha=0.2, color='navy')
    ax1.set_ylabel('E₀ (eV)', fontsize=12, fontweight='bold')
    ax1.set_title('Edge Position Trend', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sample_names, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Trend 2: SNR trend
    ax2 = axes[1]
    ax2.plot(x_pos, snr_values, 's-', linewidth=2, markersize=8, color='darkgreen', alpha=0.7)
    ax2.axhline(y=3.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Min SNR threshold')
    ax2.set_ylabel('Signal-to-Noise Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Signal Quality Trend', fontsize=14)
    ax2.set_xlabel('Sample', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sample_names, rotation=45, ha='right', fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_trends.png', dpi=300, bbox_inches='tight')
    plt.close()


def _generate_exclusion_report(all_results, output_dir: Path, config: dict | None):
    """Generate CSV report of excluded measurements based on quality results."""
    if not config:
        return None

    exclusion_cfg = (
        config.get('pipeline', {}).get('exclusion') or
        config.get('exclusion') or
        {}
    )
    if not exclusion_cfg.get('enabled', False):
        return None

    exclude_invalid = bool(exclusion_cfg.get('exclude_invalid', True))
    exclude_low_confidence = bool(exclusion_cfg.get('exclude_low_confidence', False))
    confidence_threshold = float(exclusion_cfg.get('confidence_threshold', 0.5) or 0.5)

    rows = []
    for res in all_results:
        if not res.get('success', False):
            continue
        quality = res.get('spectrum_quality', {})
        classification = quality.get('classification')
        confidence = quality.get('confidence', 1.0)
        flags = quality.get('flags', [])

        exclude = False
        reasons = []
        if exclude_invalid and classification == 'invalid':
            exclude = True
            reasons.append('invalid_classification')
        if exclude_low_confidence and confidence < confidence_threshold:
            exclude = True
            reasons.append(f'low_confidence<{confidence_threshold:.2f}')

        if exclude:
            rows.append({
                'sample_name': res.get('sample_name', ''),
                'classification': classification,
                'confidence': confidence,
                'flags': ';'.join(flags) if isinstance(flags, list) else str(flags),
                'reasons': ';'.join(reasons)
            })

    if not rows:
        return None

    quality_reports_dir = output_dir / "quality_reports" / get_timestamp_folder()
    quality_reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = quality_reports_dir / "xas_exclusion_report.csv"

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(report_path, index=False)
    print(f"Exclusion report saved to: {report_path}")
    return report_path


def _save_filtered_dataset(all_results, output_dir: Path, config: dict | None):
    """Save filtered analyzed spectra to filtered_data/ based on exclusion rules."""
    if not config:
        return None

    exclusion_cfg = (
        config.get('pipeline', {}).get('exclusion') or
        config.get('exclusion') or
        {}
    )
    if not exclusion_cfg.get('enabled', False):
        return None

    exclude_invalid = bool(exclusion_cfg.get('exclude_invalid', True))
    exclude_low_confidence = bool(exclusion_cfg.get('exclude_low_confidence', False))
    confidence_threshold = float(exclusion_cfg.get('confidence_threshold', 0.5) or 0.5)

    filtered_dir = output_dir / "filtered_data"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for res in all_results:
        if not res.get('success', False):
            continue
        quality = res.get('spectrum_quality', {})
        classification = quality.get('classification')
        confidence = quality.get('confidence', 1.0)

        exclude = False
        if exclude_invalid and classification == 'invalid':
            exclude = True
        if exclude_low_confidence and confidence < confidence_threshold:
            exclude = True

        if exclude:
            continue

        data = res.get('processed_data')
        if not data:
            continue

        # Save normalized spectrum as CSV in filtered_data
        sample_name = res.get('sample_name', 'unknown')
        output_name = sample_name.replace('_standardized', '_analyzed.csv')
        output_path = filtered_dir / output_name

        import pandas as pd
        df = pd.DataFrame({
            'energy': data.get('energy'),
            'mu_cleaned': data.get('mu_cleaned'),
            'mu_normalized': data.get('mu_normalized')
        })
        df.to_csv(output_path, index=False)
        saved += 1

    print(f"Saved {saved} filtered spectra to: {filtered_dir}")
    return filtered_dir


def _average_grouped_results(all_results, output_dir: Path, config: dict | None):
    # Average grouped spectra and save to a configured subfolder.
    if not config or not HAS_AVERAGER:
        return None

    avg_cfg = (
        config.get('pipeline', {}).get('averaging') or
        config.get('averaging') or
        {}
    )
    if not avg_cfg.get('enabled', False):
        return None

    output_subdir = avg_cfg.get('output_subdir', 'filtered_data/averaged')
    min_group_size = int(avg_cfg.get('min_group_size', 2) or 2)
    target_key = avg_cfg.get('target_key', 'mu_normalized')
    group_mode = avg_cfg.get('group_mode', 'replicate_suffix')

    exclusion_cfg = (
        config.get('pipeline', {}).get('exclusion') or
        config.get('exclusion') or
        {}
    )
    exclude_invalid = bool(exclusion_cfg.get('exclude_invalid', True))
    exclude_low_confidence = bool(exclusion_cfg.get('exclude_low_confidence', False))
    confidence_threshold = float(exclusion_cfg.get('confidence_threshold', 0.5) or 0.5)

    def _include(res):
        if not res.get('success', False):
            return False
        quality = res.get('spectrum_quality', {})
        classification = quality.get('classification')
        confidence = quality.get('confidence', 1.0)
        if exclude_invalid and classification == 'invalid':
            return False
        if exclude_low_confidence and confidence < confidence_threshold:
            return False
        return True

    groups = {}
    for res in all_results:
        if not _include(res):
            continue
        sample = res.get('sample_name', '')
        if group_mode == 'replicate_suffix':
            group_key = re.sub(r"_R\d+$", "", sample)
        else:
            group_key = sample
        groups.setdefault(group_key, []).append(res)

    if not groups:
        return None

    out_dir = output_dir / Path(output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for group, items in groups.items():
        if len(items) < min_group_size:
            continue
        energy_list = []
        mu_list = []
        for res in items:
            data = res.get('processed_data') or {}
            energy = data.get('energy')
            mu = data.get(target_key)
            if energy is None or mu is None:
                continue
            energy_list.append(energy)
            mu_list.append(mu)

        if len(energy_list) < min_group_size:
            continue

        energy_avg, mu_avg = average_spectra(energy_list, mu_list)

        import pandas as pd
        df = pd.DataFrame({
            'energy': energy_avg,
            target_key: mu_avg
        })
        out_path = out_dir / f"{group}_avg_analyzed.csv"
        df.to_csv(out_path, index=False)
        saved += 1

    print(f"Saved {saved} averaged spectra to: {out_dir}")
    return out_dir


def main():
    """Command-line interface for XAS analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='XAS Analyzer - Process and normalize standardized XAS data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all standardized files
  python main.py
  
  # Analyze specific file
  python main.py --file "JL10_1_standardized.csv"
  
  # Analyze with pattern
  python main.py --pattern "JL10*"
  
  # Custom output directory
  python main.py --output results/
  
  # Load only (no save)
  python main.py --no-save
        """
    )
    
    parser.add_argument('--file', type=str, default=None,
                       help='Specific file to analyze (relative to standardized data dir)')
    parser.add_argument('--pattern', type=str, default='*_standardized.csv',
                       help='Glob pattern for batch processing (default: "*_standardized.csv")')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: project_root/xas_results/02_analyzed_data)')
    parser.add_argument('--config', type=str, default=None,
                       help='Config YAML path (default: Tools/APS_XAS/xas_config/pipeline_config.yaml)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save analyzed files')
    parser.add_argument('--version', action='version', 
                       version=f'XAS Analyzer v{__version__}')
    
    args = parser.parse_args()
    
    save = not args.no_save
    data_dir = get_standardized_data_dir()
    config = load_pipeline_config(args.config)
    
    # Print header
    print("=" * 70)
    print(f"XAS ANALYZER v{__version__} - Normalization & Quality Assessment")
    print("=" * 70)
    print(f"\nStandardized data directory: {data_dir}")
    
    if args.file:
        # Single file analysis
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = data_dir / file_path
        
        if not file_path.exists():
            print(f"\nError: File not found: {file_path}")
            return 1
        
        print(f"\nAnalyzing file: {file_path.name}")
        print(f"Save output: {save}")
        if save:
            output_dir = Path(args.output) if args.output else get_output_dir()
            print(f"Output directory: {output_dir}")
        
        print("\nRunning analysis pipeline...")
        results = analyze_dataset(file_path, output_dir=args.output, save=save, config=config)
        
        if results['success']:
            print("\n" + "=" * 70)
            print("ANALYSIS SUMMARY")
            print("=" * 70)
            norm = results.get('normalization', {})
            norm_params = norm.get('parameters', {})
            print("\nNormalization:")
            if 'edge_step' in norm_params:
                print(f"  Edge step: {norm_params['edge_step']:.4f}")
            if 'e0' in norm_params:
                print(f"  E0: {norm_params['e0']:.2f} eV")
            if 'confidence' in norm:
                print(f"  Confidence: {norm['confidence']:.2f}")
            
            quality = results.get('spectrum_quality', {})
            print("\nQuality Assessment:")
            if 'classification' in quality:
                print(f"  Classification: {quality['classification']}")
            if 'signal_to_noise' in quality:
                print(f"  Signal-to-noise: {quality['signal_to_noise']:.2f}")
            if 'edge_jump' in quality:
                print(f"  Edge jump: {quality['edge_jump']:.4f}")
            if 'suitable_for_analysis' in quality:
                print(f"  Suitable for analysis: {quality['suitable_for_analysis']}")
        else:
            print(f"\n✗ Analysis failed: {results['error']}")
            return 1
    else:
        # Batch analysis
        print(f"\nBatch processing with pattern: {args.pattern}")
        print(f"Save output: {save}")
        if save:
            output_dir = Path(args.output) if args.output else get_output_dir()
            print(f"Output directory: {output_dir}")
        
        print("\nRunning batch analysis...")
        all_results = batch_analyze(pattern=args.pattern, 
                                   output_dir=args.output, save=save, config=config)
        
        if all_results:
            successful = sum(1 for r in all_results if r['success'])
            print("\n" + "=" * 70)
            print("BATCH ANALYSIS SUMMARY")
            print("=" * 70)
            print(f"\nTotal datasets: {len(all_results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(all_results) - successful}")
            
            # Summary statistics
            if successful > 0:
                edge_steps = [r['normalization']['parameters']['edge_step'] 
                            for r in all_results if r['success']]
                print(f"\nEdge step statistics:")
                print(f"  Mean: {np.mean(edge_steps):.4f}")
                print(f"  Std: {np.std(edge_steps):.4f}")
                print(f"  Range: {np.min(edge_steps):.4f} to {np.max(edge_steps):.4f}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
