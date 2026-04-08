"""
XAS Quality Report Plotting Module

Generate diagnostic plots from XAS quality assessment reports.
Creates scatter plots showing quality metrics across samples with thresholds
and summary statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


def _natural_sort_key(s: object):
    """Natural sorting key for sample IDs."""
    if s is None:
        return ()
    text = str(s)
    parts = __import__('re').split(r"(\d+)", text)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return tuple(key)


def _natural_sorted(iterable):
    """Return naturally sorted list."""
    return sorted(list(iterable), key=_natural_sort_key)


def plot_xas_quality_report_diagnostics(metrics_list: List,
                                       output_dir: Path,
                                       batch_name: str = "xas_quality_report",
                                       spectra_data: Optional[Dict] = None) -> Path:
    """
    Generate comprehensive quality diagnostic plots from XAS quality metrics.

    Creates a 3x3 grid with:
    - Edge jump vs SNR scatter plot
    - Quality classification distribution
    - Confidence histogram
    - Edge position distribution
    - Noise level analysis
    - Quality flags summary
    - Processing quality metrics
    - Summary statistics text box

    Parameters
    ----------
    metrics_list : List[XASSpectrumQualityMetrics]
        List of quality metrics objects
    output_dir : Path
        Output directory for plots
    batch_name : str
        Base name for plot files
    spectra_data : Dict, optional
        Optional spectra data for overlay plots

    Returns
    -------
    plot_path : Path
        Path to the generated plot
    """
    if not metrics_list:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data for plotting
    sample_ids = [m.sample_id for m in metrics_list]
    classifications = [m.classification for m in metrics_list]
    confidences = [m.confidence for m in metrics_list]
    edge_jumps = [m.edge_jump for m in metrics_list]
    snrs = [m.signal_to_noise for m in metrics_list]
    noise_levels = [m.noise_level for m in metrics_list]
    quality_flags = [m.quality_flag.value for m in metrics_list]
    data_points = [m.data_points for m in metrics_list]
    energy_ranges = [m.energy_range for m in metrics_list]

    # Sort by sample ID for consistent plotting
    sorted_pairs = sorted(enumerate(sample_ids), key=lambda x: _natural_sort_key(x[1]))
    sorted_indices = [i for i, _ in sorted_pairs]
    sample_ids = [sample_ids[i] for i in sorted_indices]
    classifications = [classifications[i] for i in sorted_indices]
    confidences = [confidences[i] for i in sorted_indices]
    edge_jumps = [edge_jumps[i] for i in sorted_indices]
    snrs = [snrs[i] for i in sorted_indices]
    noise_levels = [noise_levels[i] for i in sorted_indices]
    quality_flags = [quality_flags[i] for i in sorted_indices]
    data_points = [data_points[i] for i in sorted_indices]
    energy_ranges = [energy_ranges[i] for i in sorted_indices]

    # Create main diagnostic plot
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"XAS Quality Assessment Report - {batch_name}", fontsize=16, fontweight='bold')

    # Create 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Edge Jump vs SNR Scatter Plot
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(edge_jumps, snrs, c=confidences, cmap='viridis',
                         s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Edge Jump')
    ax1.set_ylabel('Signal-to-Noise Ratio')
    ax1.set_title('Edge Jump vs SNR')
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label('Confidence')

    # Add threshold lines
    ax1.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='Min Edge Jump')
    ax1.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Min SNR')
    ax1.legend()

    # 2. Quality Classification Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    unique_classes, counts = np.unique(classifications, return_counts=True)
    colors = {'usable': 'green', 'usable_with_warning': 'orange', 'invalid': 'red'}
    bar_colors = [colors.get(cls, 'gray') for cls in unique_classes]

    bars = ax2.bar(unique_classes, counts, color=bar_colors, alpha=0.7)
    ax2.set_title('Quality Classification')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{count}', ha='center', va='bottom')

    # 3. Confidence Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(confidences, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Confidence Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Confidence Distribution')
    ax3.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Good (>0.8)')
    ax3.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Acceptable (>0.5)')
    ax3.legend()

    # 4. Quality Flags Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    unique_flags, flag_counts = np.unique(quality_flags, return_counts=True)
    flag_colors = {'excellent': 'darkgreen', 'good': 'green', 'acceptable': 'yellow',
                  'poor': 'orange', 'invalid': 'red'}

    flag_bar_colors = [flag_colors.get(flag, 'gray') for flag in unique_flags]
    bars = ax4.bar(unique_flags, flag_counts, color=flag_bar_colors, alpha=0.7)
    ax4.set_title('Quality Flags')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, count in zip(bars, flag_counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{count}', ha='center', va='bottom')

    # 5. Data Quality Metrics
    ax5 = fig.add_subplot(gs[1, 1])
    x_pos = np.arange(len(sample_ids))

    # Plot data points and energy range
    ax5.bar(x_pos - 0.2, data_points, width=0.4, alpha=0.7, label='Data Points', color='blue')
    ax5.bar(x_pos + 0.2, energy_ranges, width=0.4, alpha=0.7, label='Energy Range (eV)', color='green')

    ax5.set_xlabel('Sample')
    ax5.set_ylabel('Count / Range')
    ax5.set_title('Data Characteristics')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([sid.split('_')[-1] for sid in sample_ids], rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Noise Analysis
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(snrs, noise_levels, c=confidences, cmap='plasma', s=60, alpha=0.7)
    ax6.set_xlabel('Signal-to-Noise Ratio')
    ax6.set_ylabel('Noise Level')
    ax6.set_title('Noise Analysis')
    ax6.grid(True, alpha=0.3)

    # Add colorbar
    cbar2 = plt.colorbar(ax6.collections[0], ax=ax6, shrink=0.8)
    cbar2.set_label('Confidence')

    # 7. Edge Jump Distribution
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(edge_jumps, bins=15, alpha=0.7, color='purple', edgecolor='black')
    ax7.set_xlabel('Edge Jump')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Edge Jump Distribution')
    ax7.axvline(x=np.mean(edge_jumps), color='red', linestyle='--',
               label=f'Mean: {np.mean(edge_jumps):.3f}')
    ax7.legend()

    # 8. SNR Distribution
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(snrs, bins=15, alpha=0.7, color='teal', edgecolor='black')
    ax8.set_xlabel('Signal-to-Noise Ratio')
    ax8.set_ylabel('Frequency')
    ax8.set_title('SNR Distribution')
    ax8.axvline(x=np.mean(snrs), color='red', linestyle='--',
               label=f'Mean: {np.mean(snrs):.1f}')
    ax8.legend()

    # 9. Summary Statistics Text Box
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    # Calculate summary statistics
    total_samples = len(metrics_list)
    usable_count = sum(1 for m in metrics_list if m.classification == 'usable')
    warning_count = sum(1 for m in metrics_list if m.classification == 'usable_with_warning')
    invalid_count = sum(1 for m in metrics_list if m.classification == 'invalid')

    avg_confidence = np.mean(confidences)
    avg_edge_jump = np.mean(edge_jumps)
    avg_snr = np.mean(snrs)

    summary_text = ".1f"".1f"".3f"".1f"".1f"".1f"f"""
SUMMARY STATISTICS

Total Samples: {total_samples}

Quality Distribution:
• Usable: {usable_count} ({usable_count/total_samples*100:.1f}%)
• Warning: {warning_count} ({warning_count/total_samples*100:.1f}%)
• Invalid: {invalid_count} ({invalid_count/total_samples*100:.1f}%)

Average Metrics:
• Confidence: {avg_confidence:.2f}
• Edge Jump: {avg_edge_jump:.3f}
• SNR: {avg_snr:.1f}

Quality Flags:
"""

    # Add quality flag breakdown
    flag_summary = {}
    for flag in quality_flags:
        flag_summary[flag] = flag_summary.get(flag, 0) + 1

    for flag, count in sorted(flag_summary.items()):
        summary_text += f"• {flag.title()}: {count}\n"

    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()

    # Save the plot
    plot_path = output_dir / f"{batch_name}_quality_diagnostics.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"XAS quality diagnostic plot saved to: {plot_path}")
    return plot_path


def plot_xas_quality_trends(metrics_list: List, output_dir: Path,
                           batch_name: str = "xas_quality_trends") -> Path:
    """
    Generate trend plots showing quality metrics evolution.

    Parameters
    ----------
    metrics_list : List[XASSpectrumQualityMetrics]
        Quality metrics
    output_dir : Path
        Output directory
    batch_name : str
        Base name for plot files

    Returns
    -------
    plot_path : Path
        Path to trend plot
    """
    if not metrics_list:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by sample ID
    sorted_metrics = sorted(metrics_list, key=lambda m: _natural_sort_key(m.sample_id))

    sample_ids = [m.sample_id for m in sorted_metrics]
    confidences = [m.confidence for m in sorted_metrics]
    edge_jumps = [m.edge_jump for m in sorted_metrics]
    snrs = [m.signal_to_noise for m in sorted_metrics]
    noise_levels = [m.noise_level for m in sorted_metrics]

    # Create trend plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("XAS Quality Metrics Trends", fontsize=14)

    x_pos = np.arange(len(sample_ids))
    x_labels = [sid.split('_')[-1] for sid in sample_ids]

    # Confidence trend
    axes[0, 0].plot(x_pos, confidences, 'o-', color='blue', linewidth=2, markersize=6)
    axes[0, 0].set_title('Confidence Trend')
    axes[0, 0].set_ylabel('Confidence')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(x_labels, rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.7)

    # Edge jump trend
    axes[0, 1].plot(x_pos, edge_jumps, 's-', color='red', linewidth=2, markersize=6)
    axes[0, 1].set_title('Edge Jump Trend')
    axes[0, 1].set_ylabel('Edge Jump')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(x_labels, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7)

    # SNR trend
    axes[1, 0].plot(x_pos, snrs, '^-', color='green', linewidth=2, markersize=6)
    axes[1, 0].set_title('Signal-to-Noise Ratio Trend')
    axes[1, 0].set_ylabel('SNR')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(x_labels, rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=10, color='orange', linestyle='--', alpha=0.7)

    # Noise level trend
    axes[1, 1].plot(x_pos, noise_levels, 'd-', color='purple', linewidth=2, markersize=6)
    axes[1, 1].set_title('Noise Level Trend')
    axes[1, 1].set_ylabel('Noise Level')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(x_labels, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f"{batch_name}_trends.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"XAS quality trends plot saved to: {plot_path}")
    return plot_path


def generate_xas_quality_report(metrics_list: List,
                               output_dir: Path,
                               batch_name: str = "xas_quality_report",
                               spectra_data: Optional[Dict] = None) -> Dict[str, Path]:
    """
    Generate complete XAS quality report with CSV and plots.

    Parameters
    ----------
    metrics_list : List[XASSpectrumQualityMetrics]
        Quality metrics for all spectra
    output_dir : Path
        Output directory
    batch_name : str
        Base name for report files
    spectra_data : Dict, optional
        Optional spectra data

    Returns
    -------
    report_files : Dict[str, Path]
        Dictionary of generated report files
    """
    try:
        from ..xas_analyzer.spectrum_quality_check import XASQualityReportGenerator
    except ImportError:
        from xas_analyzer.spectrum_quality_check import XASQualityReportGenerator

    output_dir.mkdir(parents=True, exist_ok=True)
    report_files = {}

    # Generate CSV report
    report_generator = XASQualityReportGenerator()
    for metrics in metrics_list:
        report_generator.add_quality_metrics(metrics)

    report_generator.generate_csv_report(output_dir, batch_name)
    report_generator.generate_quality_plots(output_dir, batch_name)

    # Generate diagnostic plots
    diag_plot = plot_xas_quality_report_diagnostics(
        metrics_list, output_dir, batch_name, spectra_data
    )

    # Generate trend plots
    trend_plot = plot_xas_quality_trends(metrics_list, output_dir, batch_name)

    # Collect all generated files
    report_files['csv_report'] = output_dir / f"{batch_name}.csv"
    report_files['summary'] = output_dir / f"{batch_name}_summary.txt"
    report_files['overview_plot'] = output_dir / f"{batch_name}_overview.png"
    report_files['diagnostic_plot'] = diag_plot
    report_files['trends_plot'] = trend_plot

    print(f"\nXAS Quality Report Complete:")
    print(f"CSV Report: {report_files['csv_report']}")
    print(f"Summary: {report_files['summary']}")
    print(f"Overview Plot: {report_files['overview_plot']}")
    print(f"Diagnostic Plot: {report_files['diagnostic_plot']}")
    print(f"Trends Plot: {report_files['trends_plot']}")

    return report_files