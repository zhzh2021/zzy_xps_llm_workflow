"""
XPS Plotter - Main Interface

Professional modular plotting system for XPS workflow visualization.
This module serves as the main interface, importing and exposing plotting 
functions from specialized modules organized by workflow stage.

Modules:
- data_quality: Quality control plots for extracted spectral data
- fitting: Visualization of peak fitting results and residuals  
- quantification: Atomic concentration and compositional analysis plots
- correlation: Parameter correlation and comparative analysis plots
- utils: Common utilities for configuration, styling, and file operations

Usage:
    from XPS_Plotter.plotter import plot_extracted_region_quality, plot_template_fit
    
    # Or import everything:
    from XPS_Plotter.plotter import *
"""

# ===== IMPORTS FROM SPECIALIZED MODULES =====

# Data Quality Plotting (Step 1: After data extraction)
try:
    from plot_modules.data_quality.quality_report_plots import (
        plot_extracted_region_quality,
        plot_all_extracted_regions_quality
    )
except ImportError:
    try:
        from XPS_Plotter.plot_modules.data_quality.quality_report_plots import (
            plot_extracted_region_quality,
            plot_all_extracted_regions_quality
        )
    except ImportError:
        plot_extracted_region_quality = None
        plot_all_extracted_regions_quality = None

# Fitting Plots (Step 2: Peak fitting visualization)  
try:
    from plot_modules.fitting.fitting_plots import (
        plot_template_fit,
        plot_stacked_layers_comparison
    )
except ImportError:
    try:
        from XPS_Plotter.plot_modules.fitting.fitting_plots import (
            plot_template_fit,
            plot_stacked_layers_comparison
        )
    except ImportError:
        plot_template_fit = None
        plot_stacked_layers_comparison = None

# Quantification Plots (Step 3: Compositional analysis)
try:
    from plot_modules.quantification.quantification_plots import (
        plot_atomic_concentration_per_sample,
        plot_atomic_concentration_layer_comparison,
        plot_chemistry_heatmap,
        plot_quantification_overview,
        create_quantification_plots,
        generate_component_chemistry_plots,
        generate_component_heatmap,
    )
    from plot_modules.quantification.quantification_plots_depth import (
        plot_atomic_concentration_per_sample as plot_atomic_concentration_per_sample_depth,
        plot_atomic_concentration_layer_comparison as plot_atomic_concentration_layer_comparison_depth,
        plot_chemistry_heatmap as plot_chemistry_heatmap_depth,
        plot_quantification_overview as plot_quantification_overview_depth,
        create_quantification_plots as create_quantification_plots_depth,
        generate_component_chemistry_plots as generate_component_chemistry_plots_depth,
        generate_component_heatmap as generate_component_heatmap_depth,
        plot_depth_profile_3d_waterfall,
        load_depth_profile_csv,
    )
except ImportError:
    try:
        from XPS_Plotter.plot_modules.quantification.quantification_plots import (
            plot_atomic_concentration_per_sample,
            plot_atomic_concentration_layer_comparison,
            plot_chemistry_heatmap,
            plot_quantification_overview,
            create_quantification_plots,
            generate_component_chemistry_plots,
            generate_component_heatmap,
        )
        from XPS_Plotter.plot_modules.quantification.quantification_plots_depth import (
            plot_atomic_concentration_per_sample as plot_atomic_concentration_per_sample_depth,
            plot_atomic_concentration_layer_comparison as plot_atomic_concentration_layer_comparison_depth,
            plot_chemistry_heatmap as plot_chemistry_heatmap_depth,
            plot_quantification_overview as plot_quantification_overview_depth,
            create_quantification_plots as create_quantification_plots_depth,
            generate_component_chemistry_plots as generate_component_chemistry_plots_depth,
            generate_component_heatmap as generate_component_heatmap_depth,
            plot_depth_profile_3d_waterfall,
            load_depth_profile_csv,
        )
    except ImportError:
        plot_atomic_concentration_per_sample = None
        plot_atomic_concentration_layer_comparison = None
        plot_chemistry_heatmap = None
        plot_quantification_overview = None
        create_quantification_plots = None
        generate_component_chemistry_plots = None
        generate_component_heatmap = None
        plot_atomic_concentration_per_sample_depth = None
        plot_atomic_concentration_layer_comparison_depth = None
        plot_chemistry_heatmap_depth = None
        plot_quantification_overview_depth = None
        create_quantification_plots_depth = None
        generate_component_chemistry_plots_depth = None
        generate_component_heatmap_depth = None
        plot_depth_profile_3d_waterfall = None
        load_depth_profile_csv = None

# Correlation Analysis Plots (Step 4: Advanced analysis)
try:
    from plot_modules.correlation.correlation_plots import (
        plot_correlation_matrix,
        plot_scatter_correlation,
        plot_multi_parameter_correlation,
        plot_region_comparison
    )
except ImportError:
    try:
        from XPS_Plotter.plot_modules.correlation.correlation_plots import (
            plot_correlation_matrix,
            plot_scatter_correlation,
            plot_multi_parameter_correlation,
            plot_region_comparison
        )
    except ImportError:
        plot_correlation_matrix = None
        plot_scatter_correlation = None
        plot_multi_parameter_correlation = None
        plot_region_comparison = None

# Common Utilities
try:
    from plot_modules.utils.plot_utils import (
        load_plot_config,
        save_figure_with_config,
        get_plot_colors,
        sanitize_filename,
        setup_plot_style,
        create_subplot_layout
    )
except ImportError:
    try:
        from XPS_Plotter.plot_modules.utils.plot_utils import (
            load_plot_config,
            save_figure_with_config,
            get_plot_colors,
            sanitize_filename,
            setup_plot_style,
            create_subplot_layout
        )
    except ImportError:
        load_plot_config = None
        save_figure_with_config = None
        get_plot_colors = None
        sanitize_filename = None
        setup_plot_style = None
        create_subplot_layout = None

# ===== BACKWARD COMPATIBILITY IMPORTS =====
# Import some functions that may be used by existing code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ===== CONFIGURATION =====

# Global plot configuration (loaded once)
PLOT_CONFIG = load_plot_config() if load_plot_config is not None else {}


# ===== WORKFLOW INTEGRATION FUNCTIONS =====

def generate_workflow_plots(workflow_step, data, output_dir, **kwargs):
    """
    Main function to generate plots for different workflow steps.
    
    Args:
        workflow_step (str): One of 'data_quality', 'fitting', 'quantification', 'correlation'
        data: Input data appropriate for the workflow step
        output_dir (Path): Output directory for plots
        **kwargs: Additional parameters specific to each workflow step
        
    Returns:
        dict: Results of plot generation with paths to created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if workflow_step == 'data_quality':
        if 'converted_csv_dir' in kwargs:
            return plot_all_extracted_regions_quality(
                converted_csv_dir=kwargs['converted_csv_dir'],
                plots_dir=output_dir,
                project_root=kwargs.get('project_root')
            )
        else:
            raise ValueError("data_quality step requires 'converted_csv_dir' parameter")
    
    elif workflow_step == 'fitting':
        # Batch fitting plot generation
        results = {}
        if isinstance(data, dict):
            for sample_name, fit_results in data.items():
                if isinstance(fit_results, dict):
                    for region_name, fit_result in fit_results.items():
                        vis_settings = kwargs.get('vis_settings', {})
                        plot_path = plot_template_fit(
                            sample_name=sample_name,
                            fit_result=fit_result,
                            region_name=region_name,
                            vis_settings=vis_settings,
                            outdir=str(output_dir)
                        )
                        results[f"{sample_name}_{region_name}"] = plot_path
        return results
    
    elif workflow_step == 'quantification':
        # Quantification plot generation
        results = {}
        if isinstance(data, pd.DataFrame):
            samples = data['Sample'].unique() if 'Sample' in data.columns else []
            layers = data['Layer'].unique() if 'Layer' in data.columns else []
            is_depth = 'Layer' in data.columns and len(layers) > 1

            plot_columns = kwargs.get('plot_columns', [])
            plot_labels = kwargs.get('plot_labels', plot_columns)

            per_sample_plotter = plot_atomic_concentration_per_sample_depth if is_depth and plot_atomic_concentration_per_sample_depth else plot_atomic_concentration_per_sample
            layer_plotter = plot_atomic_concentration_layer_comparison_depth if is_depth and plot_atomic_concentration_layer_comparison_depth else plot_atomic_concentration_layer_comparison
            
            for sample in samples:
                plot_path = per_sample_plotter(
                    df=data,
                    sample=sample,
                    out_dir=output_dir,
                    plot_columns=plot_columns,
                    plot_labels=plot_labels,
                )
                if plot_path:
                    results[f"sample_{sample}"] = plot_path
                    
            for layer in layers:
                plot_path = layer_plotter(
                    df=data,
                    layer=layer,
                    out_dir=output_dir,
                    plot_columns=plot_columns,
                    plot_labels=plot_labels,
                )
                if plot_path:
                    results[f"layer_{layer}"] = plot_path
                    
        return results
    
    elif workflow_step == 'correlation':
        # Correlation analysis plot generation
        results = {}
        if isinstance(data, pd.DataFrame):
            # Generate correlation matrix
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                plot_path = plot_correlation_matrix(
                    data_df=data[numeric_cols],
                    output_dir=output_dir,
                    title=kwargs.get('title', 'XPS Parameter Correlations')
                )
                results['correlation_matrix'] = plot_path
                
                # Generate multi-parameter correlation if enough parameters
                if len(numeric_cols) > 2:
                    plot_path = plot_multi_parameter_correlation(
                        data_df=data,
                        parameter_columns=numeric_cols[:6],  # Limit to avoid clutter
                        output_dir=output_dir
                    )
                    if plot_path:
                        results['multi_parameter'] = plot_path
                        
        return results
    
    else:
        raise ValueError(f"Unknown workflow step: {workflow_step}")

def _run_subdir(base_dir: Path) -> Path:
    run_id = os.environ.get("XPS_RUN_ID", "").strip()
    run_tag = ""
    if run_id:
        if "_" in run_id:
            date_part, time_part = run_id.split("_", 1)
            date_part = re.sub(r"\D", "", date_part)
            time_part = re.sub(r"\D", "", time_part)
            if len(date_part) >= 8:
                date_part = date_part[:8]
            else:
                date_part = datetime.now().strftime("%Y%m%d")
            if len(time_part) >= 6:
                time_part = time_part[:6]
            else:
                time_part = datetime.now().strftime("%H%M%S")
            run_tag = f"{date_part}{time_part}"
        else:
            digits = re.sub(r"\D", "", run_id)
            if len(digits) >= 14:
                run_tag = digits[:14]
            elif len(digits) >= 8:
                run_tag = f"{digits[:8]}{datetime.now().strftime('%H%M%S')}"
    if not run_tag:
        run_tag = datetime.now().strftime("%Y%m%d%H%M%S")
    return Path(base_dir) / run_tag


def _find_latest_run_file(root_dir: Path, filename: str) -> Optional[Path]:
    candidates = list(root_dir.glob(f"??????????????/{filename}"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.parent.name)[-1]


def _resolve_latest_run_dir(base_dir: Path) -> Path:
    run_id = os.environ.get("XPS_RUN_ID", "").strip()
    run_tag = ""
    if run_id:
        if "_" in run_id:
            date_part, time_part = run_id.split("_", 1)
            date_part = re.sub(r"\D", "", date_part)
            time_part = re.sub(r"\D", "", time_part)
            if len(date_part) >= 8:
                date_part = date_part[:8]
            else:
                date_part = datetime.now().strftime("%Y%m%d")
            if len(time_part) >= 6:
                time_part = time_part[:6]
            else:
                time_part = datetime.now().strftime("%H%M%S")
            run_tag = f"{date_part}{time_part}"
        else:
            digits = re.sub(r"\D", "", run_id)
            if len(digits) >= 14:
                run_tag = digits[:14]
    if run_tag:
        candidate = Path(base_dir) / run_tag
        if candidate.exists():
            return candidate
    if Path(base_dir).exists():
        run_dirs = [
            p for p in Path(base_dir).iterdir()
            if p.is_dir() and re.match(r"^\d{14}$", p.name)
        ]
        if run_dirs:
            return sorted(run_dirs, key=lambda p: p.name)[-1]
    return Path(base_dir)


def _generate_depth_waterfall_plots(project_root: Path, output_dir: Path) -> Dict[str, Path]:
    if plot_depth_profile_3d_waterfall is None or load_depth_profile_csv is None:
        return {}

    converted_base = project_root / "01_converted_csv"
    converted_dir = _resolve_latest_run_dir(converted_base)
    if not converted_dir.exists():
        return {}

    depth_out_dir = output_dir / "quantification" / "depth_waterfall"
    depth_out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    for region_dir in converted_dir.iterdir():
        if not region_dir.is_dir():
            continue
        if region_dir.name.lower() == "survey":
            continue
        candidates = list(region_dir.glob("aggregated_*_allHR.csv"))
        if not candidates:
            candidates = list(region_dir.glob("*_allHR.csv"))
        if not candidates:
            continue
        csv_path = sorted(candidates, key=lambda p: p.name)[-1]
        layers = load_depth_profile_csv(csv_path)
        if not layers or len(layers) < 2:
            continue
        plot_path = plot_depth_profile_3d_waterfall(
            spectra_dict=layers,
            region=region_dir.name,
            out_dir=depth_out_dir,
            config=PLOT_CONFIG if PLOT_CONFIG else None,
            depth_direction="bulk_to_surface",
        )
        if plot_path:
            results[region_dir.name] = plot_path

    return results


def _generate_workflow_status_diagram(project_root, output_dir):
    """Generate workflow status/completeness diagram."""
    from datetime import datetime
    
    config = PLOT_CONFIG.get('plot_settings', {})
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Define workflow steps
    steps = [
        "Raw Data\n(00_raw_data)",
        "Converted CSV\n(01_converted_csv)", 
        "Peak Fitting\n(02_fitted_results)",
        "Quantification\n(03_quantified_data)",
        "Plots\n(04_plots)",
        "Map Data\n(05_map_data)",
        "Correlation\n(06_correlator_results)"
    ]
    
    step_dirs = [
        project_root / "00_raw_data",
        project_root / "01_converted_csv",
        project_root / "02_fitted_results", 
        project_root / "03_quantified_data",
        project_root / "04_plots",
        project_root / "05_map_data",
        project_root / "06_correlator_results"
    ]
    
    y_pos = np.arange(len(steps))
    
    # Check which steps have data and count files
    data_availability = []
    file_counts = []
    for step_dir in step_dirs:
        if step_dir.exists():
            files = [f for f in step_dir.rglob('*') if f.is_file() and not f.name.startswith('.')]
            count = len(files)
            data_availability.append(count > 0)
            file_counts.append(count)
        else:
            data_availability.append(False)
            file_counts.append(0)
    
    # Color scheme based on file count
    colors = []
    for count in file_counts:
        if count > 50:
            colors.append('#2ecc71')  # Green - complete
        elif count > 10:
            colors.append('#f39c12')  # Orange - substantial
        elif count > 0:
            colors.append('#e74c3c')  # Red - minimal
        else:
            colors.append('#95a5a6')  # Gray - empty
    
    # Create horizontal bar chart
    ax.barh(y_pos, [1]*len(steps), color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(steps, fontsize=config.get('fonts', {}).get('axis_label_size', 14))
    ax.set_xlim([0, 1.2])
    ax.set_xlabel('Workflow Progress', fontsize=15, fontweight='bold')
    ax.set_title('XPS Workflow Status - Degree of Completeness', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks([])
    
    # Add status annotations
    for i, (count, step) in enumerate(zip(file_counts, steps)):
        if count > 50:
            status = f"✅ Complete ({count} files)"
            color = 'darkgreen'
        elif count > 10:
            status = f"🔶 Substantial ({count} files)"
            color = 'darkorange'
        elif count > 0:
            status = f"⚠️ Minimal ({count} files)"
            color = 'darkred'
        else:
            status = "❌ No Data"
            color = 'gray'
        
        ax.text(0.5, i, status, ha='center', va='center', 
               fontsize=14, fontweight='bold', color=color)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Complete (>50 files)'),
        Patch(facecolor='#f39c12', edgecolor='black', label='Substantial (11-50 files)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Minimal (1-10 files)'),
        Patch(facecolor='#95a5a6', edgecolor='black', label='Empty (0 files)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
             framealpha=0.95, edgecolor='black')
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ax.text(0.02, -0.15, f'Generated: {timestamp}', transform=ax.transAxes,
           fontsize=9, style='italic', color='gray')
    
    # Add completion percentage
    completion_pct = sum(data_availability) / len(data_availability) * 100
    ax.text(0.98, -0.15, f'Overall Completion: {completion_pct:.0f}%', 
           transform=ax.transAxes, fontsize=14,
           style='italic', color='darkblue', ha='right', fontweight='bold')
    
    plt.tight_layout()
    
    # Save as workflow_status only
    export_config = PLOT_CONFIG.get('export', {})
    output_path = output_dir / f"workflow_status.{export_config.get('default_format', 'png')}"
    plt.savefig(
        output_path, 
        dpi=config.get('dpi', 300), 
        bbox_inches=export_config.get('bbox_inches', 'tight'),
        facecolor=export_config.get('facecolor', 'white')
    )
    plt.close()
    
    return output_path


def create_workflow_overview(project_root, output_dir=None):
    """
    Create comprehensive overview plots showing the entire XPS workflow results.
    
    Args:
        project_root (Path): Project root directory
        output_dir (Path, optional): Output directory for overview plots
        
    Returns:
        dict: Paths to generated overview plots
    """
    project_root = Path(project_root)
    
    if output_dir is None:
        output_dir = _run_subdir(project_root / "04_plots" / "00_summary_dashboard")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Generate workflow status diagram
    try:
        workflow_status_path = _generate_workflow_status_diagram(project_root, output_dir)
        results['workflow_status'] = workflow_status_path
        print(f"✓ Generated workflow status: {workflow_status_path.name}")
    except Exception as e:
        print(f"⚠️  Could not generate workflow status diagram: {e}")
    
    # Generate data quality overview
    converted_csv_dir = project_root / "01_converted_csv"
    if converted_csv_dir.exists():
        quality_plots = plot_all_extracted_regions_quality(
            converted_csv_dir=converted_csv_dir,
            plots_dir=output_dir / "data_quality",
            project_root=project_root
        )
        results['data_quality'] = quality_plots
    
    # Look for quantification data - use atomic concentration file for element plots
    quant_file = project_root / "03_quantified_data" / "atomic_concentration_raw.csv"
    run_id = os.environ.get("XPS_RUN_ID", "").strip()
    if run_id and "_" in run_id:
        quant_run_dir = _run_subdir(project_root / "03_quantified_data")
        candidate = quant_run_dir / "atomic_concentration_raw.csv"
        if candidate.exists():
            quant_file = candidate
    if not quant_file.exists():
        latest = _find_latest_run_file(project_root / "03_quantified_data", "atomic_concentration_raw.csv")
        if latest is not None:
            quant_file = latest
    if quant_file.exists():
        try:
            quant_data = pd.read_csv(quant_file)
            # Use actual element columns from the file
            element_cols = [c for c in quant_data.columns if c not in ['Sample', 'Layer', 'File']]
            element_labels = [c.replace('1s', '').replace('2p', '') for c in element_cols]

            layers = quant_data['Layer'].unique() if 'Layer' in quant_data.columns else []
            is_depth = 'Layer' in quant_data.columns and len(layers) > 1
            
            quant_plots = generate_workflow_plots(
                workflow_step='quantification',
                data=quant_data,
                output_dir=output_dir / "quantification",
                plot_columns=element_cols,
                plot_labels=element_labels
            )
            results['quantification'] = quant_plots

            if is_depth:
                try:
                    depth_waterfall = _generate_depth_waterfall_plots(project_root, output_dir)
                    if depth_waterfall:
                        results['depth_waterfall'] = depth_waterfall
                except Exception as e:
                    print(f"Could not generate depth waterfall plots: {e}")

            # Generate depth-specific chemistry plots if available
            if is_depth and generate_component_chemistry_plots_depth is not None:
                try:
                    component_file = quant_file.parent / "all_components_with_atomic_percent.csv"
                    if component_file.exists():
                        comp_df = pd.read_csv(component_file)
                        depth_chem_plots = generate_component_chemistry_plots_depth(
                            comp_df,
                            output_dir / "quantification" / "chemistry_depth",
                            cfg=None,
                            config=PLOT_CONFIG if PLOT_CONFIG else None
                        )
                        results['chemistry_depth'] = depth_chem_plots
                except Exception as e:
                    print(f"Could not process depth chemistry plots: {e}")

            # Generate correlation plots
            corr_plots = generate_workflow_plots(
                workflow_step='correlation',
                data=quant_data,
                output_dir=output_dir / "correlation",
                title='XPS Atomic Concentration Correlations'
            )
            results['correlation'] = corr_plots

        except Exception as e:
            print(f"Could not process quantification data: {e}")
    
    return results


# ===== PUBLIC API EXPORTS =====

__all__ = [
    # Data Quality Plots
    'plot_extracted_region_quality',
    'plot_all_extracted_regions_quality',
    
    # Fitting Plots
    'plot_template_fit', 
    'plot_stacked_layers_comparison',
    
    # Quantification Plots
    'plot_atomic_concentration_per_sample',
    'plot_atomic_concentration_layer_comparison',
    'plot_chemistry_heatmap',
    'plot_quantification_overview',
    'generate_component_chemistry_plots',
    'generate_component_heatmap',
    'plot_atomic_concentration_per_sample_depth',
    'plot_atomic_concentration_layer_comparison_depth',
    'plot_chemistry_heatmap_depth',
    'plot_quantification_overview_depth',
    'create_quantification_plots_depth',
    'generate_component_chemistry_plots_depth',
    'generate_component_heatmap_depth',
    'plot_depth_profile_3d_waterfall',
    'load_depth_profile_csv',
    
    # Correlation Plots
    'plot_correlation_matrix',
    'plot_scatter_correlation', 
    'plot_multi_parameter_correlation',
    'plot_region_comparison',
    
    # Utilities
    'load_plot_config',
    'save_figure_with_config',
    'get_plot_colors',
    'sanitize_filename',
    'setup_plot_style',
    'create_subplot_layout',
    
    # Workflow Integration
    'generate_workflow_plots',
    'create_workflow_overview',
    
    # Configuration
    'PLOT_CONFIG'
]


# ===== INITIALIZATION =====

# Setup matplotlib style using configuration
if setup_plot_style is not None and PLOT_CONFIG:
    setup_plot_style(PLOT_CONFIG)
    print(f"XPS Plotter initialized with modular structure")
    print(f"Available workflow steps: data_quality, fitting, quantification, correlation")
    print(f"Configuration loaded from: {PLOT_CONFIG.get('_config_path', 'defaults')}")


# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    """Main execution to generate all XPS plots."""
    import sys
    from pathlib import Path
    
    # Determine project root
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1])
    else:
        # Default to project_root relative to this script
        project_root = Path(__file__).resolve().parents[2] / "project_root"
    
    if not project_root.exists():
        print(f"❌ Project root not found: {project_root}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("🎨 XPS PLOTTER - Generating All Plots")
    print("="*70)
    print(f"📂 Project Root: {project_root}")
    print()
    
    # Generate workflow overview (includes workflow status diagram)
    print("📊 Generating workflow overview and status diagram...")
    try:
        results = create_workflow_overview(project_root)
        print(f"✅ Workflow overview generated")
        if 'workflow_status' in results:
            print(f"   📈 Workflow status: {results['workflow_status']}")
    except Exception as e:
        print(f"⚠️  Error generating workflow overview: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ XPS Plotter Complete")
    print("="*70)
    print(f"📁 Output directory: {project_root / '04_plots'}")
    print()
