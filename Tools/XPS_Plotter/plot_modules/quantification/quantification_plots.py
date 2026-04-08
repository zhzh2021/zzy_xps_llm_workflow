"""
Quantification Plots Module

Functions for visualizing XPS quantification results, including atomic 
concentrations, layer comparisons, and compositional analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
from math import ceil
import re


def _natural_sort_key(s: object):
    """Key function for natural (human) sorting of strings with numbers.

    Splits input on digit groups and converts digit parts to integers so that
    names like S2 come before S10.
    """
    if s is None:
        return ()
    text = str(s)
    parts = re.split(r"(\d+)", text)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return tuple(key)


def _natural_sorted(iterable):
    return sorted(list(iterable), key=_natural_sort_key)

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


def _format_sample_label(sample_name: str) -> str:
    """Normalize sample labels while preserving uniqueness."""
    stem = Path(str(sample_name)).stem or str(sample_name)
    match = re.match(r"(.+?)_L\d+$", stem)
    if match:
        stem = match.group(1)
    return stem


def _sample_layer_label(sample_name: str, layer: Optional[object]) -> str:
    """Create consistent sample-layer labels for heatmaps."""
    base = _format_sample_label(sample_name)
    if layer is None or (isinstance(layer, float) and np.isnan(layer)):
        return base
    return f"{base}_L{int(layer)}"


def _detect_element_columns(
    df: pd.DataFrame,
    explicit_columns: Optional[List[str]] = None,
) -> tuple[List[str], List[str]]:
    """Determine which columns represent element atomic percentages."""
    if explicit_columns:
        cols = [col for col in explicit_columns if col in df.columns]
    else:
        # First try to find columns ending with _at%
        cols = [c for c in df.columns if isinstance(c, str) and c.lower().endswith("_at%")]
        if not cols:
            # Fallback: find numeric columns that look like XPS regions (e.g., C1s, O1s, F1s)
            exclude = {
                "Sample",
                "Layer",
                "Region",
                "Component",
                "Component_atomic_percent",
                "Element_atomic_percent",
                "Center_eV",
                "FWHM_eV",
                "Eta_mix",
                "Total_area",
                "Area_counts",
                "Area_percent",
                "File",
            }
            numeric_cols = [
                c for c in df.columns
                if c not in exclude
                and pd.api.types.is_numeric_dtype(df[c])
            ]
            cols = numeric_cols

    # Create clean labels - remove region suffixes like "1s", "2p", etc.
    labels = []
    for c in cols:
        if isinstance(c, str):
            if c.lower().endswith("_at%"):
                labels.append(c[:-4])
            else:
                # Remove XPS orbital notation (e.g., "C1s" -> "C", "P2p" -> "P")
                label = re.sub(r'[0-9]+[spdf]$', '', c)
                labels.append(label if label else c)
        else:
            labels.append(str(c))
    
    return cols, labels


def plot_atomic_concentration_per_sample(
    df: pd.DataFrame,
    sample: str,
    out_dir: Path,
    plot_columns: Optional[List[str]] = None,
    plot_labels: Optional[List[str]] = None,
    config=None,
):
    """Plot atomic concentration for a specific sample across layers."""
    if config is None:
        config = load_plot_config()

    if not plot_columns:
        plot_columns, inferred_labels = _detect_element_columns(df)
        plot_labels = inferred_labels
    elif not plot_labels:
        _, inferred_labels = _detect_element_columns(df, plot_columns)
        plot_labels = inferred_labels

    if not plot_columns:
        return None
    
    sub = df[df["Sample"] == sample]
    if sub.empty:
        return None
    if plot_columns:
        sub = sub[sub[plot_columns].notna().any(axis=1)]
        if sub.empty:
            return None

    layers = _natural_sorted(sub["Layer"].unique().tolist())
    n_elements = len(plot_columns)
    n_layers = len(layers)

    plot_config = config['plot_settings']
    fig, ax = plt.subplots(figsize=tuple(plot_config['figure_sizes']['single_plot']))
    indices = np.arange(n_elements)

    width = 0.5 / max(n_layers, 1)
    colors = sns.color_palette("Set2", n_layers)

    for l_idx, layer in enumerate(layers):
        row = sub[sub["Layer"] == layer]
        if row.empty:
            vals = [np.nan] * n_elements
        else:
            row_vals = row.iloc[0]
            vals = [float(row_vals[col]) if pd.notna(row_vals[col]) else np.nan for col in plot_columns]
        ax.bar(indices + l_idx * width, vals, width=width, color=colors[l_idx], label=f"Layer {layer}")

    font_config = plot_config['fonts']
    ax.set_xticks(indices + (n_layers - 1) * width / 2)
    ax.set_xticklabels(plot_labels, rotation=45, ha="right", fontsize=font_config['axis_label_size'])
    ax.set_ylabel("Atomic Concentration (%)", fontsize=font_config['axis_label_size'])
    ax.tick_params(axis="x", labelsize=font_config['tick_label_size'])
    ax.tick_params(axis="y", labelsize=font_config['tick_label_size'])
    ax.set_title(f"{sample}: Atomic Concentration by Layer", fontsize=font_config['title_size'], fontweight='bold')
    ax.grid(axis="y", linestyle="--", alpha=plot_config['lines']['grid_alpha'])
    ax.legend(ncol=2, fontsize=font_config['legend_size'], frameon=False)
    fig.tight_layout()

    fname = out_dir / f"{sanitize_filename(sample)}_atomic_concentration.{config['export']['default_format']}"
    fig.savefig(fname, dpi=plot_config['dpi'])
    plt.close(fig)
    return fname


def plot_atomic_concentration_layer_comparison(
    df: pd.DataFrame,
    layer: int,
    out_dir: Path,
    plot_columns: Optional[List[str]] = None,
    plot_labels: Optional[List[str]] = None,
    config=None,
):
    """Plot atomic concentration comparison for a specific layer across samples. This is for depth profile only with multi layers."""
    if config is None:
        config = load_plot_config()
    
    if not plot_columns:
        plot_columns, inferred_labels = _detect_element_columns(df)
        plot_labels = inferred_labels
    elif not plot_labels:
        _, inferred_labels = _detect_element_columns(df, plot_columns)
        plot_labels = inferred_labels

    if not plot_columns:
        return None

    sub = df[df["Layer"] == layer]
    if plot_columns:
        sub = sub[sub[plot_columns].notna().any(axis=1)]
    if sub.empty:
        return None

    samples = sub["Sample"].unique().tolist()
    samples_sorted = _natural_sorted(samples)

    n_elements = len(plot_columns)
    n_samples = len(samples_sorted)
    width = 0.8 / max(n_samples, 1)

    plot_config = config['plot_settings']
    fig, ax = plt.subplots(figsize=tuple(plot_config['figure_sizes']['single_plot']))
    indices = np.arange(n_elements)
    colors = sns.color_palette("pastel", n_samples)

    for s_idx, sample in enumerate(samples_sorted):
        row = sub[sub["Sample"] == sample]
        if row.empty:
            vals = [np.nan] * n_elements
        else:
            row_vals = row.iloc[0]
            vals = [float(row_vals[col]) if pd.notna(row_vals[col]) else np.nan for col in plot_columns]
        ax.bar(indices + s_idx * width, vals, width=width, color=colors[s_idx], label=sample)

    font_config = plot_config['fonts']
    ax.set_xticks(indices + (n_samples - 1) * width / 2)
    ax.set_xticklabels(plot_labels, rotation=45, ha="right", fontsize=font_config['axis_label_size'])
    ax.set_ylabel("Atomic Concentration (%)", fontsize=font_config['axis_label_size'])
    ax.tick_params(axis="x", labelsize=font_config['tick_label_size'])
    ax.tick_params(axis="y", labelsize=font_config['tick_label_size'])
    ax.set_title(f"Layer {layer}: Atomic Concentration Comparison", fontsize=font_config['title_size'], fontweight='bold')
    ax.grid(axis="y", linestyle="--", alpha=plot_config['lines']['grid_alpha'])
    ax.legend(ncol=2, fontsize=font_config['legend_size'], frameon=False)
    fig.tight_layout()

    fname = out_dir / f"layer_{layer}_atomic_concentration_comparison.{config['export']['default_format']}"
    fig.savefig(fname, dpi=plot_config['dpi'])
    plt.close(fig)
    return fname


def plot_chemistry_heatmap(chemistry_name, component_data_dict, plots_dir, config=None):
    """
    Create heatmap showing distribution of specific chemistry components.
    
    Args:
        chemistry_name (str): Name of the chemistry (e.g., "C1s", "O1s")
        component_data_dict (dict): Component data organized by sample/layer
        plots_dir (Path): Output directory for plots
        config (dict, optional): Plot configuration
        
    Returns:
        Path: Path to saved heatmap plot
    """
    if config is None:
        config = load_plot_config()
    
    plot_config = config['plot_settings']
    
    # Convert component data to DataFrame for heatmap
    heatmap_data = pd.DataFrame(component_data_dict).fillna(0)

    # Reorder columns (samples) and index (components) naturally for readability
    try:
        natural_cols = _natural_sorted(heatmap_data.columns.tolist())
        heatmap_data = heatmap_data.reindex(columns=natural_cols)
    except Exception:
        pass
    try:
        natural_index = _natural_sorted(heatmap_data.index.tolist())
        heatmap_data = heatmap_data.reindex(index=natural_index)
    except Exception:
        pass
    
    # Transpose to show samples on Y-axis and components on X-axis
    heatmap_data = heatmap_data.T
    
    if heatmap_data.empty:
        return None

    fig, ax = plt.subplots(figsize=tuple(plot_config['figure_sizes']['comparison_plot']))
    
    # Create heatmap with custom styling
    sns.heatmap(
        heatmap_data,
        cmap='coolwarm',
        annot=False,
        fmt='.1f',
        cbar_kws={'label': 'Atomic %'},
        ax=ax,
        vmin=0,
        vmax=heatmap_data.values.max() * 0.8,
    )
    
    font_config = plot_config['fonts']
    ax.set_title(f'{chemistry_name} Component Distribution', 
                fontsize=font_config['title_size'], fontweight='bold')
    ax.set_xlabel('Sample_Layer', fontsize=font_config['axis_label_size'], fontweight='bold')
    ax.set_ylabel('Component', fontsize=font_config['axis_label_size'], fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = plots_dir / f'{chemistry_name}_chemistry_heatmap.{config["export"]["default_format"]}'
    plt.savefig(
        plot_path, 
        dpi=plot_config['dpi'], 
        bbox_inches=config['export']['bbox_inches'],
        facecolor=config['export']['facecolor']
    )
    plt.close(fig)
    return plot_path


def generate_component_heatmap(
    results_df: pd.DataFrame,
    plots_dir: Path,
    cfg,
    chemistry_name: str,
    config=None,
) -> Optional[Path]:
    """
    High-level helper mirroring legacy API used by XPS_Quantifier.
    """
    if config is None:
        config = load_plot_config()

    if results_df is None or results_df.empty:
        return None

    chemistry_groups = None
    if cfg is not None and hasattr(cfg, 'chemistry') and getattr(cfg.chemistry, 'groups', None):
        chemistry_groups = cfg.chemistry.groups
    elif isinstance(cfg, dict):
        chemistry_groups = cfg.get('chemistry_groups')

    if not chemistry_groups or chemistry_name not in chemistry_groups:
        return None

    chem_group = chemistry_groups[chemistry_name]
    region_prefix = getattr(chem_group, 'region_prefix', None)
    if not region_prefix:
        return None

    chem_df = results_df[
        (results_df['Region'].str.startswith(str(region_prefix)))
        & (~results_df['Component_atomic_percent'].isna())
    ].copy()
    if chem_df.empty:
        return None

    chem_df['Sample_Layer'] = chem_df.apply(
        lambda row: _sample_layer_label(row.get('Sample', ''), row.get('Layer', None)),
        axis=1
    )

    component_dict: Dict[str, Dict[str, float]] = {}
    for row in chem_df.itertuples():
        component_dict.setdefault(row.Component, {})[row.Sample_Layer] = row.Component_atomic_percent

    if not component_dict:
        return None

    return plot_chemistry_heatmap(
        chemistry_name=chemistry_name,
        component_data_dict=component_dict,
        plots_dir=plots_dir,
        config=config,
    )


def plot_quantification_overview(quantification_results, plots_dir, config=None):
    """
    Create overview plot of all quantification results.
    
    Args:
        quantification_results (dict): Complete quantification data
        plots_dir (Path): Output directory for plots
        config (dict, optional): Plot configuration
        
    Returns:
        Path: Path to saved overview plot
    """
    if config is None:
        config = load_plot_config()
    
    plot_config = config['plot_settings']
    
    # Create summary figure with multiple subplots
    fig = plt.figure(figsize=tuple(plot_config['figure_sizes']['summary_plot']))
    
    # This would contain overview plots of the quantification results
    # Implementation depends on the structure of quantification_results
    
    font_config = plot_config['fonts']
    fig.suptitle('XPS Quantification Overview', 
                fontsize=font_config['title_size'], fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = plots_dir / f'quantification_overview.{config["export"]["default_format"]}'
    plt.savefig(
        plot_path, 
        dpi=plot_config['dpi'], 
        bbox_inches=config['export']['bbox_inches'],
        facecolor=config['export']['facecolor']
    )
    plt.close(fig)
    return plot_path


def create_quantification_plots(data_df: pd.DataFrame, output_dir: Path, config=None):
    """Create all quantification plots."""
    if config is None:
        config = load_plot_config()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_plots = []
    
    print("Generating quantification plots...")
    
    # Overview plot
    overview_plot = plot_quantification_overview(data_df, output_dir, config)
    if overview_plot:
        generated_plots.append(overview_plot)
    
    if not data_df.empty:
        plot_columns, plot_labels = _detect_element_columns(data_df)

        samples = _natural_sorted(data_df['Sample'].unique()) if 'Sample' in data_df.columns else []
        for sample in samples:
            plot_path = plot_atomic_concentration_per_sample(
                df=data_df,
                sample=sample,
                out_dir=output_dir,
                plot_columns=plot_columns,
                plot_labels=plot_labels,
                config=config
            )
            if plot_path:
                generated_plots.append(plot_path)
        
        layers = _natural_sorted(data_df['Layer'].unique()) if 'Layer' in data_df.columns else []
        for layer in layers:
            plot_path = plot_atomic_concentration_layer_comparison(
                df=data_df,
                layer=layer,
                out_dir=output_dir,
                plot_columns=plot_columns,
                plot_labels=plot_labels,
                config=config
            )
            if plot_path:
                generated_plots.append(plot_path)

        for element_col, element_label in zip(plot_columns, plot_labels):
            plot_path = plot_element_cross_sample_distribution(
                df=data_df,
                element_col=element_col,
                element_label=element_label,
                out_dir=output_dir,
                config=config,
            )
            if plot_path:
                generated_plots.append(plot_path)
    
    print(f"Generated {len(generated_plots)} quantification plots")
    return generated_plots


def plot_element_cross_sample_distribution(
    df: pd.DataFrame,
    element_col: str,
    element_label: str,
    out_dir: Path,
    config=None,
) -> Optional[Path]:
    """Create a cross-sample bar chart for a specific element."""
    if config is None:
        config = load_plot_config()

    if element_col not in df.columns:
        return None

    data = df[df[element_col].notna()].copy()
    if data.empty:
        return None

    samples = _natural_sorted(data["Sample"].unique())
    layers = _natural_sorted(data["Layer"].unique()) if "Layer" in data.columns else [None]

    plot_config = config['plot_settings']
    fig, ax = plt.subplots(figsize=tuple(plot_config['figure_sizes']['comparison_plot']))

    x_pos = np.arange(len(samples))
    width = 0.8 / max(len(layers), 1)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(layers))]

    for idx, layer in enumerate(layers):
        layer_data = data if layer is None else data[data["Layer"] == layer]
        values = []
        for sample in samples:
            sample_data = layer_data[layer_data["Sample"] == sample]
            if not sample_data.empty:
                values.append(float(sample_data[element_col].iloc[0]))
            else:
                values.append(0.0)

        legend_label = f"Layer {layer}" if layer is not None else "All Layers"
        ax.bar(
            x_pos + idx * width,
            values,
            width=width,
            color=colors[idx],
            edgecolor='black',
            label=legend_label,
        )

    font_config = plot_config['fonts']
    ax.set_xticks(x_pos + (len(layers) - 1) * width / 2)
    ax.set_xticklabels(
        [_format_sample_label(sample) for sample in samples],
        rotation=45,
        ha="right",
        fontsize=font_config['tick_label_size'],
    )
    ax.set_ylabel("Atomic Concentration (%)", fontsize=font_config['axis_label_size'])
    ax.set_title(
        f"{element_label}: Atomic Concentration Across Samples",
        fontsize=font_config['title_size'],
        fontweight='bold',
    )
    ax.grid(axis="y", linestyle="--", alpha=plot_config['lines']['grid_alpha'])
    if len(layers) > 1:
        ax.legend(ncol=2, fontsize=font_config['legend_size'], frameon=False)
    fig.tight_layout()

    fname = out_dir / f"{sanitize_filename(element_label)}_cross_sample_concentration.{config['export']['default_format']}"
    fig.savefig(
        fname,
        dpi=plot_config['dpi'],
        bbox_inches=config['export']['bbox_inches'],
        facecolor=config['export']['facecolor'],
    )
    plt.close(fig)
    return fname


def generate_component_chemistry_plots(
    results_df: pd.DataFrame,
    plots_dir: Path,
    cfg=None,
    config=None,
) -> List[Path]:
    """Create cross-sample component plots using configured chemistry groups."""
    if config is None:
        config = load_plot_config()
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if results_df is None or results_df.empty:
        return []

    df = results_df.dropna(subset=['Component_atomic_percent']).copy()
    if df.empty:
        return []

    chemistry_groups = None
    if cfg is not None and hasattr(cfg, 'chemistry') and getattr(cfg.chemistry, 'groups', None):
        chemistry_groups = cfg.chemistry.groups
    elif isinstance(cfg, dict):
        chemistry_groups = cfg.get('chemistry_groups')

    if not chemistry_groups:
        return []

    plot_settings = config['plot_settings']
    export_config = config['export']
    dpi = getattr(getattr(cfg, 'plot', None), 'dpi', plot_settings['dpi'])
    font_cfg = plot_settings['fonts']

    generated_paths: List[Path] = []

    for chem_name, chem_group in chemistry_groups.items():
        region_prefix = getattr(chem_group, 'region_prefix', None)
        if not region_prefix:
            continue

        chem_df = df[df['Region'].str.startswith(str(region_prefix))].copy()
        if chem_df.empty:
            continue

        # Treat each layer as a distinct sample label for cross-sample distribution
        chem_df['Sample_Label'] = chem_df.apply(
            lambda row: _sample_layer_label(row.get('Sample'), row.get('Layer')),
            axis=1,
        )
        pivot = chem_df.pivot_table(
            index='Sample_Label',
            columns='Component',
            values='Component_atomic_percent',
            aggfunc='mean',
            fill_value=0,
        )
        if pivot.empty:
            continue

        # Reorder pivot rows (samples) naturally for consistent plotting
        try:
            natural_index = _natural_sorted(pivot.index.tolist())
            pivot = pivot.reindex(natural_index)
        except Exception:
            pass

        # Get chemistry-specific colormap from config
        chemistry_colormaps = config.get('chemistry_colormaps', {})
        cmap = chemistry_colormaps.get(chem_name, chemistry_colormaps.get('default', 'Pastel2'))

        fig, ax = plt.subplots(figsize=tuple(plot_settings['figure_sizes']['comparison_plot']))
        # Create stacked bar chart with softer appearance
        bars = pivot.plot(
            kind='bar',
            ax=ax,
            stacked=True,
            width=0.6,  # Narrower bars for cleaner look
            edgecolor='gray',  # Softer edge color
            linewidth=0.4,
            colormap=cmap,
            alpha=0.85,  # Subtle transparency for softer colors
        )

        ax.set_xlabel('Sample', fontsize=font_cfg['axis_label_size'], fontweight='bold')
        ax.set_ylabel('Component Atomic %', fontsize=font_cfg['axis_label_size'], fontweight='bold')
        ax.set_title(
            f'{chem_name}: Component Distribution Across Samples',
            fontsize=font_cfg['title_size'],
            fontweight='bold',
        )
        ax.grid(axis='y', linestyle='--', alpha=plot_settings['lines']['grid_alpha'])
        ax.set_xticklabels(
            pivot.index.tolist(),
            rotation=45,
            ha='right',
            fontsize=font_cfg['tick_label_size'],
        )
        ax.tick_params(axis='x', labelsize=font_cfg['tick_label_size'])
        ax.tick_params(axis='y', labelsize=font_cfg['tick_label_size'])
        ax.legend(
            title='Component',
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            fontsize=font_cfg['legend_size'],
            frameon=False,
        )

        totals = pivot.sum(axis=1)
        for idx, (sample, total) in enumerate(totals.items()):
            if total > 0:
                ax.text(
                    idx,
                    total + 0.2,
                    f'{total:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=font_cfg['tick_label_size'],
                    fontweight='bold',
                )

        fig.tight_layout(rect=[0, 0.15, 1, 1])
        cross_sample_path = plots_dir / f"{sanitize_filename(chem_name)}_cross_sample_components.{export_config['default_format']}"
        fig.savefig(
            cross_sample_path,
            dpi=dpi,
            bbox_inches=export_config['bbox_inches'],
            facecolor=export_config['facecolor'],
        )
        plt.close(fig)
        generated_paths.append(cross_sample_path)

        trend_plot = _create_individual_component_trends(
            chem_df,
            chem_name,
            plots_dir,
            plot_settings,
            export_config,
            dpi,
        )
        if trend_plot:
            generated_paths.append(trend_plot)

    return generated_paths


def _create_individual_component_trends(
    chem_df: pd.DataFrame,
    chem_name: str,
    plots_dir: Path,
    plot_settings: Dict,
    export_config: Dict,
    dpi: int,
) -> Optional[Path]:
    """Generate per-component trend plots across samples."""
    components = _natural_sorted(chem_df['Component'].dropna().unique().tolist())
    if not components:
        return None

    chem_df = chem_df.copy()
    # Treat each layer as a distinct sample label to show depth trends
    chem_df['Sample_Label'] = chem_df.apply(
        lambda row: _sample_layer_label(row.get('Sample'), row.get('Layer')),
        axis=1,
    )

    n_components = len(components)
    n_cols = min(3, n_components)  # 3 columns layout
    n_rows = ceil(n_components / n_cols)
    base_w, base_h = plot_settings['figure_sizes']['comparison_plot']
    # Adjust figure size for 3-column layout
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(base_w * 1.5, max(base_h * 1.0, base_h * n_rows / 2.0)),
    )
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    font_cfg = plot_settings['fonts']

    for idx, component in enumerate(components):
        ax = axes[idx]
        comp_df = chem_df[chem_df['Component'] == component]
        if comp_df.empty:
            ax.set_visible(False)
            continue

        sample_avg = comp_df.groupby('Sample_Label')['Component_atomic_percent'].mean()
        # Adaptive bar width based on number of samples
        n_samples = len(sample_avg)
        if n_samples <= 2:
            bar_width = 0.2
        elif n_samples <= 4:
            bar_width = 0.5
        else:
            bar_width = 0.7
        sample_avg.plot(kind='bar', ax=ax, color='#4C72B0', alpha=0.85, width=bar_width)
        ax.set_title(component, fontsize=font_cfg['axis_label_size']-1, fontweight='bold', pad=8)
        ax.set_ylabel('Atomic %', fontsize=font_cfg['axis_label_size']-2)
        ax.set_xlabel('Samples', fontsize=font_cfg['axis_label_size']-2)
        ax.grid(axis='y', alpha=plot_settings['lines']['grid_alpha'], linewidth=0.5)
        ax.set_xticklabels(
            sample_avg.index.tolist(),
            rotation=45,
            ha='right',
            fontsize=font_cfg['tick_label_size']-1,
        )
        ax.tick_params(axis='y', labelsize=font_cfg['tick_label_size']-1)

        # Only show value labels for bars with significant values (> 1%)
        for bar_idx, (sample, value) in enumerate(sample_avg.items()):
            if value > 1.0:  # Only label significant values
                ax.text(
                    bar_idx,
                    value + max(value * 0.05, 0.2),  # Dynamic spacing based on value
                    f'{value:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=font_cfg['tick_label_size']-2,
                )
        
        # Adjust y-axis limit to accommodate labels
        if len(sample_avg) > 0:
            max_value = sample_avg.max()
            ax.set_ylim(0, max_value * 1.15)  # Add 15% headroom for labels

    for idx in range(n_components, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f'{chem_name}: Individual Component Trends',
        fontsize=font_cfg['title_size'],
        fontweight='bold',
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.98], h_pad=3.0, w_pad=2.5)  # Extra bottom margin for long labels

    plot_path = plots_dir / f"{sanitize_filename(chem_name)}_component_trends.{export_config['default_format']}"
    fig.savefig(
        plot_path,
        dpi=dpi,
        bbox_inches=export_config['bbox_inches'],
        facecolor=export_config['facecolor'],
    )
    plt.close(fig)
    return plot_path


# 3D Waterfall plot now in separate module: depth_3d_waterfall.py
# Import and use like this:
#   from .depth_3d_waterfall import plot_depth_profile_3d_waterfall, load_depth_profile_csv
