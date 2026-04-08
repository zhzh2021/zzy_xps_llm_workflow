"""
Fitting Plots Module

Functions for visualizing XPS peak fitting results, including fitted curves,
residuals, and component analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

# Import utilities - handle both relative and absolute imports
try:
    from plot_modules.utils.plot_utils import load_plot_config, sanitize_filename
except ImportError:
    try:
        from ..utils.plot_utils import load_plot_config, sanitize_filename
    except ImportError:
        # Fallback for standalone usage
        import sys
        from pathlib import Path
        utils_path = Path(__file__).parent.parent / "utils"
        sys.path.insert(0, str(utils_path))
        from plot_utils import load_plot_config, sanitize_filename


def plot_template_fit(sample_name,
                      fit_result,
                      region_name,
                      vis_settings: Dict,
                      outdir: str = "plots",
                      figsize=None,
                      dpi: int | None = None):
    """Create publication-quality plot showing template-based fit."""
    
    # Load plot configuration
    config = load_plot_config()
    plot_config = config['plot_settings']
    
    # 1) Build color lookup
    comp_colors = vis_settings.get('component_colors', {}) or {}
    cmap_name = vis_settings.get('type', 'viridis')
    reverse = bool(vis_settings.get('reverse', False))
    if not comp_colors:
        cmap = plt.get_cmap(cmap_name)
        names = list(fit_result["components"].keys())
        vals = np.linspace(0, 1, len(names))
        if reverse:
            vals = vals[::-1]
        comp_colors = {n: cmap(v) for n, v in zip(names, vals)}

    # Use configuration for figure size and DPI
    if figsize is None:
        figsize_to_use = tuple(plot_config['figure_sizes']['single_plot'])
    else:
        figsize_to_use = tuple(figsize)
    if dpi is None:
        dpi_to_use = plot_config['dpi']
    else:
        dpi_to_use = int(dpi)

    # Use configuration for height ratios
    height_ratios = [3, 1]  # Main plot:residuals ratio
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_to_use, height_ratios=height_ratios)

    x = fit_result["x"]
    # Safe access for optional keys when fit_result may not be a dict
    def _safe_get(obj, key, default=None):
        return obj.get(key, default) if isinstance(obj, dict) else default

    bg_label = f"{_safe_get(fit_result, 'background_type', 'Shirley').title()} background"

    # Main plot - NO STATISTICS HERE
    line_width = plot_config['lines']['linewidth']
    fit_alpha = 0.8
    comp_alpha = 0.6
    
    ax1.plot(x, fit_result["raw"], 'k-', linewidth=1.5, label="Raw data", alpha=0.8)
    ax1.plot(x, fit_result["baseline"], 'gray', linewidth=1.5, label=bg_label)
    ax1.plot(x, fit_result["corrected"], 'b-', linewidth=1.5, label="Background corrected")

    # Individual components with template names - plotted on background-corrected data
    for name, curve in fit_result["components"].items():
        col = comp_colors.get(name, 'gray')
        ax1.fill_between(
            x,
            0,
            curve,
            color=col,
            alpha=comp_alpha,
            label=name,
        )

    # Total fit - red solid line (fits the background-corrected data)
    ax1.plot(
        x,
        fit_result["fit"],
        color='red',
        linestyle='-',
        linewidth=line_width,
        alpha=fit_alpha,
        label="Total fit",
    )

    # Formatting with config font sizes
    font_config = plot_config['fonts']
    ax1.set_ylabel("Intensity", fontsize=font_config['axis_label_size'], fontweight='bold')
    ax1.set_title(
        f"{sample_name} - {region_name} (Template: {fit_result['template_used']})",
        fontsize=font_config['title_size'],
        fontweight='bold'
    )
    
    # Legend inside plot box
    ax1.legend(
        loc='best',
        fontsize=font_config['legend_size'],
        framealpha=0.9
    )
    ax1.grid(True, alpha=plot_config['lines']['grid_alpha'])
    ax1.invert_xaxis()

    # Apply tick label sizes from config for both axes
    tick_size = font_config.get('tick_label_size', 12)
    ax1.tick_params(axis='both', labelsize=tick_size)

    # Residuals plot
    residuals = fit_result["corrected"] - fit_result["fit"]
    ax2.plot(x, residuals, color=plot_config['colors']['secondary'], linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Binding Energy (eV)", fontsize=font_config['axis_label_size'], fontweight='bold')
    ax2.set_ylabel("Residuals", fontsize=font_config['axis_label_size'], fontweight='bold')
    ax2.grid(True, alpha=plot_config['lines']['grid_alpha'])
    ax2.invert_xaxis()
    ax2.tick_params(axis='both', labelsize=tick_size)

    # Add fit statistics to RESIDUALS plot with config font size
    total_area = sum(p["area"] for p in fit_result["peaks"] if np.isfinite(p["area"]))
    textstr = (
        f"R² = {fit_result['r2']:.3f}\nSignificant Components (≥2%): {len(fit_result['peaks'])}\n"
    )
    textstr += f"Total Area: {total_area:.0f}\n"
    textstr += "─" * 15 + "\n"

    for p in fit_result["peaks"]:
        area_pct = 100 * p["area"] / total_area if total_area > 0 else 0
        textstr += f"{p['name']}: {area_pct:.1f}%\n"

    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax2.text(
        0.02,
        0.98,
        textstr,
        transform=ax2.transAxes,
        fontsize=font_config['info_text_size'],
        verticalalignment='top',
        bbox=props,
    )

    # Apply tight layout
    plt.tight_layout()

    # Save plot with region tag using config export settings
    export_config = config['export']
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sample_tag = sanitize_filename(sample_name)
    region_tag = sanitize_filename(region_name)
    outpath = outdir / f"{sample_tag}_{region_tag}_template_fit.{export_config['default_format']}"
    
    fig.savefig(
        outpath, 
        dpi=dpi_to_use, 
        bbox_inches=export_config['bbox_inches'],
        facecolor=export_config['facecolor'],
        transparent=export_config['transparent']
    )
    plt.close(fig)

    return outpath


def plot_stacked_layers_comparison(
    sample_name: str,
    all_layer_fits: dict,  # {label: fit_result}
    region_name: str,
    vis_settings: Dict,
    template_path: Path,
    outdir: str = "plots",
    figsize=None,
    dpi: int | None = None,
) -> Path:
    """
    Stack each layer/sample in its own subplot. Use consistent component colors
    for all subplots based on vis_settings.

    Handles:
      - Single or multiple layers/samples
      - Different x-data ranges across layers (plots common region)

    Styles:
      - Corrected (data): dotted line with optional markers
      - Total Fit: solid line
    """

    if not all_layer_fits:
        raise ValueError("No layer fits provided")
        
    # Load plot configuration
    config = load_plot_config()
    plot_config = config['plot_settings']

    # Style configuration (with sensible defaults)
    corrected_color = vis_settings.get('corrected_color', 'b')  # blue
    corrected_ls = vis_settings.get('corrected_linestyle', 'None')
    corrected_marker = vis_settings.get('corrected_marker', 'o')  # or None
    corrected_msize = float(vis_settings.get('corrected_markersize', 4.0))
    corrected_alpha = float(vis_settings.get('corrected_alpha', 0.9))
    corrected_markevery = vis_settings.get('corrected_markevery', None)  # int or None

    # Use config for fit styling
    fit_color = vis_settings.get('fit_color', plot_config['colors']['primary'])
    fit_ls = vis_settings.get('fit_linestyle', '-')  # solid
    fit_lw = float(vis_settings.get('fit_linewidth', plot_config['lines']['linewidth']))
    fit_alpha = float(vis_settings.get('fit_alpha', 0.8))

    # Natural sort for numeric sample names (e.g., S1, S2, ..., S10, S11)
    def natural_sort_key(item):
        """Sort key that handles numeric parts correctly (S1, S2, ..., S10, S11)"""
        import re
        label = str(item[0])
        # Split into text and number parts
        parts = re.split(r'(\d+)', label)
        # Convert numeric parts to integers for proper sorting
        return [int(p) if p.isdigit() else p.lower() for p in parts]
    
    layers = sorted(all_layer_fits.items(), key=natural_sort_key)
    n = len(layers)

    # Figure sizing and DPI from config
    if dpi is None:
        dpi_to_use = plot_config['dpi']
    else:
        dpi_to_use = int(dpi)

    # Handle single vs multi-layer figure size using config
    if n == 1:
        # Use single stacked plot size for individual layers
        single_stacked = plot_config['figure_sizes'].get('single_stacked', [10, 8])
        figsize_default = tuple(single_stacked)
        figsize_to_use = tuple(figsize) if figsize is not None else figsize_default
        fig, axes = plt.subplots(1, 1, figsize=figsize_to_use)
        axes = [axes]
    else:
        # Use multi-layer base size - tall and narrow for stacked plots
        base_width, base_height = plot_config['figure_sizes'].get('multi_stacked_base', [4, 3])
        # Scale height for number of layers
        figsize_default = (base_width, base_height * n)
        print(f"[DEBUG] Multi-layer stacked: base=[{base_width}, {base_height}], n={n}, final figsize={figsize_default}")
        figsize_to_use = tuple(figsize) if figsize is not None else figsize_default
        fig, axes = plt.subplots(n, 1, figsize=figsize_to_use, sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = [axes]

    # Collect all unique components across all layers (natural-sorted)
    all_comps = sorted(
        {c for _, fit in layers for c in fit["components"].keys()},
        key=lambda s: [int(p) if p.isdigit() else p.lower() for p in __import__('re').split(r"(\d+)", str(s))],
    )
    comp_colors = vis_settings.get('component_colors', {}) or {}
    cmap_name = vis_settings.get('type', 'viridis')
    reverse = bool(vis_settings.get('reverse', False))
    missing = [c for c in all_comps if c not in comp_colors]
    
    # Only use matplotlib colormap if type is not 'custom'
    if missing and cmap_name.lower() != 'custom':
        cmap = plt.get_cmap(cmap_name)
        vals = np.linspace(0, 1, len(missing))
        if reverse:
            vals = vals[::-1]
        comp_colors.update({c: cmap(v) for c, v in zip(missing, vals)})
    elif missing and cmap_name.lower() == 'custom':
        # For custom colormap with missing components, use default colors
        default_colors = ['#87CEEB', '#FFB6D9', '#B19CD9', '#FFA07A', '#98D8C8', '#FFB347', '#F0E68C', '#FFCCCB']
        for i, comp in enumerate(missing):
            comp_colors[comp] = default_colors[i % len(default_colors)]

    # Find common x-range across all layers
    x_mins = []
    x_maxs = []
    for _, fit in layers:
        x = fit["x"]
        x_mins.append(x.min())
        x_maxs.append(x.max())

    # Common range is the intersection of all ranges
    common_x_min = max(x_mins)  # highest minimum
    common_x_max = min(x_maxs)  # lowest maximum

    # Validate common range exists
    if common_x_min >= common_x_max:
        print(f"   Warning: No overlapping x-range found across layers")
        print(f"      Using full range instead")
        common_x_min = min(x_mins)
        common_x_max = max(x_maxs)
    else:
        print(f"   Common x-range: [{common_x_min:.2f}, {common_x_max:.2f}] eV")

    # Find common y-range across all layers (using common x-range data)
    y_mins = []
    y_maxs = []
    for _, fit in layers:
        x = fit["x"]
        mask = (x >= common_x_min) & (x <= common_x_max)
        corrected = fit["corrected"][mask]
        if len(corrected) > 0:
            y_mins.append(corrected.min())
            y_maxs.append(corrected.max())
    
    common_y_min = min(y_mins) if y_mins else 0
    common_y_max = max(y_maxs) if y_maxs else 1
    # Add small padding (5%)
    y_range = common_y_max - common_y_min
    common_y_min -= y_range * 0.05
    common_y_max += y_range * 0.05
    print(f"   Common y-range: [{common_y_min:.0f}, {common_y_max:.0f}] intensity")

    for ax, (label, fit) in zip(axes, layers):
        x = fit["x"]

        # Filter data to common x-range
        mask = (x >= common_x_min) & (x <= common_x_max)
        x_plot = x[mask]
        corrected_plot = fit["corrected"][mask]
        fit_plot = fit["fit"][mask]

        if len(x_plot) < 2:
            print(f"   Warning: Layer '{label}' has insufficient data in common range")
            continue

        # Corrected spectrum (dotted + optional markers)
        me = corrected_markevery
        if me is None and len(x_plot) > 0:
            me = max(1, len(x_plot) // 50)  # roughly 50 markers across the trace

        ax.plot(
            x_plot,
            corrected_plot,
            color=corrected_color,
            linestyle=corrected_ls,
            marker=corrected_marker if corrected_marker else None,
            markevery=me if corrected_marker else None,
            ms=corrected_msize if corrected_marker else None,
            lw=1.5,
            alpha=corrected_alpha,
            label="Corrected",
        )

        # Components (areas) - also filter to common range
        for name, curve in fit["components"].items():
            curve_plot = curve[mask]
            ax.fill_between(
                x_plot,
                0,
                curve_plot,
                color=comp_colors.get(name, 'gray'),
                alpha=0.6,
                label=name,
            )

        # Total Fit (solid line)
        ax.plot(
            x_plot,
            fit_plot,
            color=fit_color,
            linestyle=fit_ls,
            lw=fit_lw,
            alpha=fit_alpha,
            label="Total Fit",
        )

        # Add layer/sample name as annotation using config fonts
        font_config = plot_config['fonts']
        ax.text(
            0.03,
            0.97,
            str(label),
            transform=ax.transAxes,
            fontsize=font_config['subtitle_size'],
            fontweight='bold',
            verticalalignment='top',
            horizontalalignment='left',
        )

        ax.set_ylabel("Intensity", fontsize=font_config['axis_label_size'], fontweight='bold')
        ax.grid(alpha=plot_config['lines']['grid_alpha'])

    # Set x-axis label using config fonts
    axes[-1].set_xlabel("Binding Energy (eV)", fontsize=font_config['axis_label_size'], fontweight='bold')

    # Set common x-limits and y-limits for all subplots
    for ax in axes:
        ax.set_xlim(common_x_max, common_x_min)  # high → low (binding energy)
        ax.set_ylim(common_y_min, common_y_max)  # common intensity scale
        ax.tick_params(labelsize=font_config['tick_label_size'])
        ax.grid(True, alpha=plot_config['lines']['grid_alpha'])

    # Legend outside plot box for stacked comparison
    legend_config = plot_config.get('legend', {})
    stacked_bbox_anchor = legend_config.get('stacked_bbox_anchor', [1.01, 1.0])
    stacked_position = legend_config.get('stacked_position', 'upper left')
    
    axes[0].legend(
        loc=stacked_position, 
        bbox_to_anchor=stacked_bbox_anchor, 
        fontsize=font_config['legend_size']
    )

    # Adjust title for single vs multiple layers
    if n == 1:
        title = f"{region_name}"
    else:
        title = f"{region_name} - Comparison"

    axes[0].set_title(
        title, 
        fontsize=font_config['title_size'], 
        fontweight='bold'
    )

    # Apply layout adjustments from config (remove subplot gaps, padding)
    layout_config = plot_config.get('layout', {}).get('subplot_adjust', {})
    if layout_config:
        fig.subplots_adjust(**layout_config)
    else:
        # Add spacing between subplots for multi-layer plots
        if n > 1:
            fig.tight_layout(rect=[0, 0, 0.95, 1], h_pad=2.0)
        else:
            fig.tight_layout()

    # Save figure using config export settings
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    export_config = config['export']
    fn = f"{sample_name}_{region_name}_stacked.{export_config['default_format']}"
    out = outdir / fn
    
    # Save with config export settings
    fig.savefig(
        out, 
        dpi=dpi_to_use, 
        bbox_inches=export_config['bbox_inches'],
        facecolor=export_config['facecolor'],
        transparent=export_config['transparent']
    )
    plt.close(fig)

    return out
