"""
Quality Report Plotting Module

Generate diagnostic plots from Stage 2 quality assessment reports.
Creates scatter plots showing quality metrics across samples with thresholds
and summary statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional


def _natural_sort_key(s: object):
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
    return sorted(list(iterable), key=_natural_sort_key)


_KNOWN_EXTENSIONS = (
    ".spe",
    ".vgd",
    ".npl",
    ".xy",
    ".txt",
    ".asc",
    ".dat",
    ".csv",
    ".vms",
    ".vamas",
    ".pro",
)


def _strip_known_extension(name: str) -> str:
    if not name:
        return name
    lower = name.lower()
    for ext in _KNOWN_EXTENSIONS:
        if lower.endswith(ext):
            return name[:-len(ext)]
    return name


def plot_quality_report_diagnostics(metrics_list: List, region: str, 
                                     output_dir: Path, config: Dict,
                                     spectra_dict: Optional[Dict] = None,
                                     plots_dir: Optional[Path] = None) -> Path:
    """
    Generate comprehensive quality diagnostic plots from quality metrics.
    
    Creates a 3x3 grid with:
    - Raw spectra overlay plot (BE vs intensity)
    - SNR scatter plot with thresholds
    - Peak height scatter plot  
    - Peak FWHM scatter plot
    - Resolution bar plot
    - Quality distribution pie chart
    - Summary text box
    
    Args:
        metrics_list: List of SpectrumQualityMetrics objects
        region: Region name (e.g., 'C1s')
        output_dir: Output directory for plots
        config: Configuration dict with thresholds
        spectra_dict: Dict of {key: Spectrum} for raw data overlay
        
    Returns:
        Path to saved plot file
    """
    if not metrics_list:
        return None
    
    # Create figure with expanded grid to include raw overlay
    has_raw_data = spectra_dict is not None and len(spectra_dict) > 0
    
    if has_raw_data:
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.35)
    else:
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    
    fig.suptitle(f'Quality Assessment Report: {region}', 
                 fontsize=18, fontweight='bold')
    
    # Plot 0: Raw spectra overlay (if data available)
    if has_raw_data:
        ax0 = fig.add_subplot(gs[0, :])
        
        # Generate colors for each spectrum
        colors_raw = plt.cm.tab20(np.linspace(0, 1, len(spectra_dict)))
        
        for idx, (spec_key, spectrum) in enumerate(spectra_dict.items()):
            try:
                energy = spectrum.energy
                intensity = spectrum.intensity
                
                # Extract sample_id for legend
                sample_id = spec_key or getattr(spectrum, 'name', '')
                if hasattr(spectrum, 'metadata'):
                    metadata = spectrum.metadata
                    original_name = metadata.get('original_spectrum', '')
                    layer_index = metadata.get('layer_index')
                    is_depth = bool(metadata.get('depth_profile', False))

                    if is_depth and layer_index:
                        base_name = original_name or sample_id
                        base_name = _strip_known_extension(str(base_name))
                        layer_tag = f"L{layer_index}"
                        if layer_tag not in base_name:
                            sample_id = f"{base_name}_{layer_tag}"
                        else:
                            sample_id = base_name
                    elif original_name:
                        sample_id = original_name

                if region:
                    suffix = f"_{region}"
                    if str(sample_id).endswith(suffix):
                        sample_id = str(sample_id)[:-len(suffix)]

                sample_id = _strip_known_extension(str(sample_id))
                
                # Plot with color and label
                ax0.plot(energy, intensity, 
                        color=colors_raw[idx], 
                        linewidth=1.8, 
                        alpha=0.8, 
                        label=sample_id)
            except Exception as e:
                print(f"[PlotQuality] Warning: Failed to plot {spec_key}: {e}")
                continue
        
        ax0.set_xlabel('Binding Energy (eV)', fontsize=14, fontweight='bold')
        ax0.set_ylabel('Intensity (a.u.)', fontsize=14, fontweight='bold')
        ax0.set_title(f'Raw Extracted Spectra: {region}', fontsize=15, fontweight='bold')
        ax0.tick_params(labelsize=11)
        ax0.grid(True, alpha=0.3, linestyle=':')
        ax0.invert_xaxis()  # XPS convention: high BE on left
        
        # Add legend with smaller font if many spectra, place outside to avoid truncation
        n_spec = len(spectra_dict)
        legend_ncol = 2 if n_spec > 6 else 1
        legend_fontsize = 14 if n_spec > 8 else 12
        ax0.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),
            fontsize=legend_fontsize,
            ncol=legend_ncol,
            framealpha=0.9,
            borderaxespad=0.0
        )
    
    # Adjust subplot indices if raw data plot is present
    row_offset = 1 if has_raw_data else 0
    
    # Sort metrics_list by sample_id naturally for consistent ordering
    try:
        metrics_sorted = sorted(metrics_list, key=lambda m: _natural_sort_key(getattr(m, 'sample_id', None)))
    except Exception:
        metrics_sorted = list(metrics_list)

    # Extract data
    sample_ids = [m.sample_id for m in metrics_sorted]
    n_samples = len(sample_ids)
    x_positions = np.arange(n_samples)

    snr_xps = np.array([m.snr_xps for m in metrics_sorted])
    peak_heights = np.array([m.peak_height for m in metrics_sorted])
    peak_to_baseline = np.array([m.peak_to_baseline_ratio for m in metrics_sorted])
    peak_fwhm = np.array([m.peak_width_fwhm for m in metrics_sorted])
    resolution = np.array([m.points_per_ev for m in metrics_sorted])
    quality_flags = [m.quality_flag.value for m in metrics_sorted]
    
    # Color mapping based on quality (lighter shades)
    color_map = {
        'excellent': '#7dcea0',    # Light green
        'good': '#85c1e9',         # Light blue  
        'acceptable': '#f8c471',   # Light orange
        'poor': '#ec7063',         # Light red
        'suspicious': '#e59866'    # Light dark red
    }
    colors = [color_map.get(flag, 'lightgray') for flag in quality_flags]
    
    # --- Plot 1: SNR scatter plot with thresholds ---
    ax1 = fig.add_subplot(gs[row_offset, 0])
    ax1.scatter(x_positions, snr_xps, c=colors, s=120, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    # Add threshold lines
    ax1.axhline(config['min_snr_xps_excellent'], color='#7dcea0', 
                linestyle='--', linewidth=2.5, label='Excellent', alpha=0.8)
    ax1.axhline(config['min_snr_xps_good'], color='#85c1e9', 
                linestyle='--', linewidth=2.5, label='Good', alpha=0.8)
    ax1.axhline(config['min_snr_xps_acceptable'], color='#f8c471', 
                linestyle='--', linewidth=2.5, label='Acceptable', alpha=0.8)
    
    ax1.set_xlabel('Sample ID', fontsize=14, fontweight='bold')
    ax1.set_ylabel('XPS SNR', fontsize=14, fontweight='bold')
    ax1.set_title('Signal-to-Noise Ratio', fontsize=15, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(sample_ids, rotation=45, ha='right', fontsize=11)
    ax1.tick_params(axis='both', labelsize=12)
    
    # --- Plot 2: Peak-to-Baseline Ratio scatter plot ---
    ax2 = fig.add_subplot(gs[row_offset, 1])
    ax2.scatter(x_positions, peak_to_baseline, c=colors, s=120, alpha=0.8, 
                edgecolors='black', linewidth=1.5)

    # Add threshold line at 0.2 for quantification suitability
    ax2.axhline(0.2, color='#e74c3c', linestyle='--',
                linewidth=2.5, label='Threshold: 0.2', alpha=0.8)

    ax2.set_xlabel('Sample ID', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Peak-to-Baseline Ratio', fontsize=14, fontweight='bold')
    ax2.set_title('P/B Ratio', fontsize=15, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(sample_ids, rotation=45, ha='right', fontsize=11)
    ax2.tick_params(axis='both', labelsize=12)
    
    # --- Plot 3: Peak FWHM scatter plot ---
    ax3 = fig.add_subplot(gs[row_offset+1, 0])
    ax3.scatter(x_positions, peak_fwhm, c=colors, s=120, alpha=0.8, 
                edgecolors='black', linewidth=1.5)
    
    # Add median line
    median_fwhm = np.median(peak_fwhm[peak_fwhm > 0])  # Exclude zeros
    ax3.axhline(median_fwhm, color='#af7ac5', linestyle='--', 
                linewidth=2.5, label=f'Median: {median_fwhm:.2f} eV', alpha=0.8)
    
    ax3.set_xlabel('Sample ID', fontsize=14, fontweight='bold')
    ax3.set_ylabel('FWHM (eV)', fontsize=14, fontweight='bold')
    ax3.set_title('Peak Width', fontsize=15, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(sample_ids, rotation=45, ha='right', fontsize=11)
    ax3.tick_params(axis='both', labelsize=12)
    
    # --- Plot 4: Summary Statistics Text Box ---
    ax4 = fig.add_subplot(gs[row_offset, 2])
    ax4.axis('off')
    
    # Calculate statistics
    quality_counts = {}
    for flag in quality_flags:
        quality_counts[flag] = quality_counts.get(flag, 0) + 1
    
    hr_count = sum(1 for m in metrics_list if m.is_hr_scan)
    shift_count = sum(1 for m in metrics_list if m.suspected_shift)
    fitting_count = sum(1 for m in metrics_list if m.suitable_for_fitting)
    
    # Count low quality (poor + suspicious)
    low_quality_count = quality_counts.get('poor', 0) + quality_counts.get('suspicious', 0)
    
    # Build summary text
    summary_lines = [
        f'📊 QUALITY SUMMARY',
        f'══════════════════════',
        f'Total Spectra: {n_samples}',
        f'⚠️  Low Quality: {low_quality_count}/{n_samples} spectra',
        f'',
        f'SNR Statistics:',
        f'  Mean: {np.mean(snr_xps):.2f}',
        f'  Median: {np.median(snr_xps):.2f}',
        f'  Range: {np.min(snr_xps):.2f} - {np.max(snr_xps):.2f}',
        f'',
        f'Scan Characteristics:',
        f'  ✓ HR Scans: {hr_count}/{n_samples}',
        f'  ⚡ Suspected Shifts: {shift_count}/{n_samples}',
        f'  📈 Suitable for Fitting: {fitting_count}/{n_samples}',
    ]
    
    summary_text = '\n'.join(summary_lines)
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=13, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4, pad=1))
    
    # --- Plot 5: Resolution bar plot ---
    ax5 = fig.add_subplot(gs[row_offset+1, 1])
    bars = ax5.bar(x_positions, resolution, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    ax5.axhline(config['hr_resolution_threshold'], color="#857dce", 
                linestyle='--', linewidth=2.5, label='HR threshold', alpha=0.8)
    
    ax5.set_xlabel('Sample ID', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Points per eV', fontsize=14, fontweight='bold')
    ax5.set_title('Spectral Resolution', fontsize=15, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=12)
    ax5.grid(axis='y', alpha=0.3)
    ax5.set_xticks(x_positions)
    ax5.set_xticklabels(sample_ids, rotation=45, ha='right', fontsize=11)
    ax5.tick_params(axis='both', labelsize=12)
    
    # --- Plot 6: Quality Distribution Pie Chart ---
    # Place in row 2, column 2 (rightmost position)
    ax6 = fig.add_subplot(gs[row_offset+1, 2])
    
    pie_colors = [color_map.get(k, 'lightgray') for k in quality_counts.keys()]
    wedges, texts, autotexts = ax6.pie(
        quality_counts.values(), 
        labels=[f"{k}\n({v})" for k, v in quality_counts.items()],
        autopct='%1.0f%%', 
        colors=pie_colors, 
        startangle=90,
        textprops={'fontsize': 12}
    )
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax6.set_title('Quality Distribution', fontsize=15, fontweight='bold')
    
    if has_raw_data:
        # Leave room for the legend outside the plot
        plt.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        plt.tight_layout()
    
    # Save plot to 04_plots/01_converted_csv (timestamped per run if provided)
    output_dir = Path(output_dir)
    if plots_dir is None:
        plots_dir = output_dir.parent.parent / "04_plots" / "01_converted_csv"
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"{region}_quality_diagnostics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_sample_quality_details(metrics_list: List, region: str, 
                                output_dir: Path, config: Dict,
                                max_samples_per_plot: int = 20,
                                plots_dir: Optional[Path] = None) -> List[Path]:
    """
    Generate detailed per-sample quality plots showing all metrics.
    
    If there are many samples, creates multiple plots.
    
    Args:
        metrics_list: List of SpectrumQualityMetrics objects
        region: Region name
        output_dir: Output directory
        config: Configuration dict
        max_samples_per_plot: Maximum samples to show per plot
        
    Returns:
        List of paths to saved plot files
    """
    if not metrics_list:
        return []
    
    n_samples = len(metrics_list)
    n_plots = (n_samples + max_samples_per_plot - 1) // max_samples_per_plot
    
    plot_paths = []
    
    for plot_idx in range(n_plots):
        start_idx = plot_idx * max_samples_per_plot
        end_idx = min(start_idx + max_samples_per_plot, n_samples)
        
        subset = metrics_list[start_idx:end_idx]
        try:
            subset = sorted(subset, key=lambda m: _natural_sort_key(getattr(m, 'sample_id', None)))
        except Exception:
            subset = list(subset)
        sample_ids = [m.sample_id for m in subset]
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle(f'{region} - Sample Quality Details',
                     fontsize=14, fontweight='bold')
        
        # Color by quality (lighter shades)
        color_map = {
            'excellent': '#7dcea0',    # Light green
            'good': '#85c1e9',         # Light blue  
            'acceptable': '#f8c471',   # Light orange
            'poor': '#ec7063',         # Light red
            'suspicious': '#e59866'    # Light dark red
        }
        colors = [color_map.get(m.quality_flag.value, 'lightgray') for m in subset]
        
        x_pos = np.arange(len(sample_ids))
        
        # Plot 1: SNR + Peak-to-baseline ratio
        ax = axes[0]
        ax2 = ax.twinx()
        
        snr_bars = ax.bar(x_pos - 0.2, [m.snr_xps for m in subset], 
                          width=0.2, color=colors, alpha=0.8, label='SNR')
        ratio_bars = ax2.bar(x_pos + 0.2, [m.peak_to_baseline_ratio for m in subset],
                            width=0.2, color='#af7ac5', alpha=0.6, label='PBR')

        ax.set_ylabel('Signal-to-Noise Ratio', fontsize=15, fontweight='bold')
        ax2.set_ylabel('Peak-to-Baseline Ratio', fontsize=15, fontweight='bold')
        ax.set_title('Signal Quality Metrics', fontsize=15, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sample_ids, rotation=45, ha='right', fontsize=15)
        ax.tick_params(axis='both', labelsize=15)
        ax2.tick_params(axis='y', labelsize=15)
        ax.grid(axis='y', alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=15)
        
        # Plot 2: Peak characteristics
        ax = axes[1]
        ax2 = ax.twinx()
        
        height_bars = ax.bar(x_pos - 0.2, [m.peak_height for m in subset],
                            width=0.2, color=colors, alpha=0.8, label='Height')
        fwhm_bars = ax2.bar(x_pos + 0.2, [m.peak_width_fwhm for m in subset],
                           width=0.2, color='#f5b7b1', alpha=0.6, label='FWHM')
        
        ax.set_ylabel('Peak Height', fontsize=15, fontweight='bold')
        ax2.set_ylabel('FWHM (eV)', fontsize=15, fontweight='bold')
        ax.set_title('Peak Characteristics', fontsize=15, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sample_ids, rotation=45, ha='right', fontsize=15)
        ax.tick_params(axis='both', labelsize=15)
        ax2.tick_params(axis='y', labelsize=15)
        ax.grid(axis='y', alpha=0.3)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=15)
        
        # Plot 3: Noise characteristics
        ax = axes[2]
        ax2 = ax.twinx()
        
        baseline_bars = ax.bar(x_pos - 0.2, [m.baseline_std for m in subset],
                              width=0.2, color=colors, alpha=0.8, label='Baseline Std')
        noise_bars = ax2.bar(x_pos + 0.2, [m.relative_noise for m in subset],
                            width=0.2, color='#f1948a', alpha=0.6, label='Rel. Noise')

        ax.set_ylabel('Baseline Std Dev', fontsize=15, fontweight='bold')
        ax2.set_ylabel('Relative Noise', fontsize=15, fontweight='bold')
        ax.set_title('Noise Characteristics', fontsize=15, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sample_ids, rotation=45, ha='right', fontsize=15)
        ax.tick_params(axis='both', labelsize=15)
        ax2.tick_params(axis='y', labelsize=15)
        ax.grid(axis='y', alpha=0.3)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=15)
        
        plt.tight_layout()
        
        # Save to 04_plots/01_converted_csv (timestamped per run if provided)
        if plots_dir is None:
            plots_dir = output_dir.parent.parent / "04_plots" / "01_converted_csv"
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / f"{region}_quality_details_{plot_idx+1}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        plot_paths.append(plot_path)
    
    return plot_paths
