from __future__ import annotations
from pathlib import Path
import re
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shutil
import json
import yaml
from datetime import datetime


def sanitize_filename(s: str) -> str:
    """Keep alphanumerics, _, ., -, replace others with underscore."""
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s))


# ========== PLOT CONFIGURATION LOADER ==========
def load_plot_config(config_path=None):
    """
    Load plot configuration from YAML file.
    Returns default settings if config file not found.
    """
    default_config = {
        'plot_settings': {
            'figure_sizes': {
                'single_plot': [10, 6],
                'overview_plot': [12, 8],
                'summary_plot': [15, 10],
                'single_stacked': [8, 5],
                'multi_stacked_base': [6, 3],
                'correlation_plot': [10, 8]
            },
            'dpi': 300,
            'fonts': {
                'title_size': 16,
                'subtitle_size': 14,
                'axis_label_size': 14,
                'axis_label_size_large': 15,
                'tick_label_size': 16,
                'legend_size': 18,
                'annotation_size': 15,
                'residual_label_size': 10,
                'info_text_size': 9
            },
            'legend': {
                'position': 'upper left',
                'bbox_anchor': [1.05, 1.0],
                'stacked_position': 'upper left',
                'stacked_bbox_anchor': [1.01, 1.0],
                'alpha': 1.0,
                'frameon': True
            },
            'lines': {
                'fit_line_width': 2,
                'fit_alpha': 0.8,
                'component_alpha': 0.6,
                'grid_alpha': 0.3
            },
            'layout': {
                'subplot_adjust': {
                    'top': 0.95,
                    'bottom': 0.05,
                    'left': 0.10,
                    'right': 0.95,
                    'hspace': 0.0,
                    'wspace': 0.2
                },
                'title_pad': 15
            },
            'height_ratios': {
                'main_residual': [3, 1]
            },
            'colors': {
                'fit_line': 'red',
                'data_points': 'black',
                'residuals': 'blue',
                'grid': 'gray'
            }
        },
        'export': {
            'formats': ['png', 'pdf'],
            'default_format': 'png',
            'bbox_inches': 'tight',
            'transparent': False,
            'facecolor': 'white'
        }
    }
    
    if config_path is None:
        # Try to find config file in common locations
        possible_paths = [
            Path.cwd() / "xps_config" / "plot_settings.yaml",
            Path.cwd() / "plot_settings.yaml",
            Path(__file__).parent.parent.parent / "project_root" / "xps_config" / "plot_settings.yaml"
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                
            # Merge user config with defaults (user config takes precedence)
            def merge_dicts(default, user):
                result = default.copy()
                for key, value in user.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = merge_dicts(result[key], value)
                    else:
                        result[key] = value
                return result
            
            config = merge_dicts(default_config, user_config)
            print(f"Loaded plot configuration from: {config_path}")
            return config
            
        except Exception as e:
            print(f"Error loading plot config from {config_path}: {e}")
            print("   Using default settings")
    else:
        print("Using default plot settings (no config file found)")
    
    return default_config

# Global plot configuration
PLOT_CONFIG = load_plot_config()


# ========== XPS WORKFLOW INTEGRATION ==========
try:
    import sys
    from pathlib import Path
    
    # Add Tools directory to path for import
    tools_dir = Path(__file__).resolve().parents[1]
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    
    from xps_workflow_manager import get_workflow_config, update_module_paths
    WORKFLOW_MANAGER_AVAILABLE = True
except ImportError:
    WORKFLOW_MANAGER_AVAILABLE = False
    print("⚠️  Workflow manager not available, using legacy paths")

def resolve_plot_paths(project_root=None):
    """
    Resolve plotter paths using workflow manager for unified structure.
    """
    if WORKFLOW_MANAGER_AVAILABLE:
        try:
            workflow_config = get_workflow_config(project_root)
            plotter_config = update_module_paths('plotter', workflow_config)
            
            return {
                "project_root": workflow_config.project_root,
                "input_dirs": {
                    "raw_data": workflow_config.raw_data_dir,
                    "fitted_data": workflow_config.fits_output_dir,
                    "quantified_data": workflow_config.quant_output_dir,
                    "correlator_data": workflow_config.correlator_output_dir
                },
                "output_dir": workflow_config.plots_output_dir,
            }
        except Exception as e:
            print(f"⚠️  Workflow manager error: {e}, falling back to legacy paths")
    
    # Legacy fallback
    root = Path(project_root or Path.cwd())
    return {
        "project_root": root,
        "input_dirs": {
            "raw_data": root / "00_raw_data",
            "fitted_data": root / "02_fitted_results", 
            "quantified_data": root / "03_quantified_data",
            "correlator_data": root / "05_correlator_results"
        },
        "output_dir": root / "04_plots",
    }

class XPSPlotOrganizer:
    """
    Professional XPS Plot Organization System
    
    Implements dual organization:
    - Plots remain in source step folders for traceability
    - Organized copies/links in centralized 04_plots for accessibility
    """
    
    def __init__(self, project_root=None):
        """Initialize plot organizer with project configuration."""
        self.paths = resolve_plot_paths(project_root)
        self.output_dir = self.paths["output_dir"]
        self.project_root = self.paths["project_root"]
        
        # Create organized plot structure
        self.plot_categories = {
            "01_data_conversion": "Raw data conversion and validation plots",
            "02_peak_fitting": "Peak fitting results and diagnostics", 
            "03_quantification": "Elemental and component quantification",
            "04_correlation_analysis": "ML correlation and trend analysis",
            "00_summary_dashboard": "High-level overview and summary plots"
        }
        
        self.setup_directory_structure()
        
    def setup_directory_structure(self):
        """Create organized plot directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create category directories
        for category, description in self.plot_categories.items():
            category_dir = self.output_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # Create README for each category
            readme_path = category_dir / "README.md"
            if not readme_path.exists():
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {category.replace('_', ' ').title()}\n\n")
                    f.write(f"{description}\n\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Create master index
        self.create_master_index()
    
    def create_master_index(self):
        """Create master plot index file."""
        index_path = self.output_dir / "PLOT_INDEX.md"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("# XPS Analysis Plot Index\n\n")
            f.write("This directory contains organized plots from all XPS workflow steps.\n\n")
            f.write("## Directory Structure\n\n")
            
            for category, description in self.plot_categories.items():
                f.write(f"### {category}/\n")
                f.write(f"{description}\n\n")
            
            f.write("## Workflow Integration\n\n")
            f.write("Plots are organized from source directories:\n")
            f.write(f"- **Raw Data**: `{self.paths['input_dirs']['raw_data'].name}/`\n")
            f.write(f"- **Fitted Results**: `{self.paths['input_dirs']['fitted_data'].name}/plots/`\n") 
            f.write(f"- **Quantified Data**: `{self.paths['input_dirs']['quantified_data'].name}/plots/`\n")
            f.write(f"- **Correlation Results**: `{self.paths['input_dirs']['correlator_data'].name}/`\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def organize_fitting_plots(self):
        """Organize peak fitting plots into centralized structure."""
        print("\n📊 Organizing peak fitting plots...")
        
        fitting_plots_src = self.paths["input_dirs"]["fitted_data"] / "plots"
        if not fitting_plots_src.exists():
            print(f"  ⚠️ No fitting plots found at {fitting_plots_src}")
            return
            
        fitting_dest = self.output_dir / "02_peak_fitting"
        
        # Copy individual layer plots
        individual_src = fitting_plots_src / "individual_layers"
        if individual_src.exists():
            individual_dest = fitting_dest / "individual_fits"
            individual_dest.mkdir(exist_ok=True)
            self._copy_plots_with_manifest(individual_src, individual_dest, "Individual peak fitting results")
        
        # Copy stacked comparison plots  
        stacked_src = fitting_plots_src / "stacked_comparison"
        if stacked_src.exists():
            stacked_dest = fitting_dest / "comparative_analysis"
            stacked_dest.mkdir(exist_ok=True)
            self._copy_plots_with_manifest(stacked_src, stacked_dest, "Multi-sample comparison plots")
    
    def organize_quantification_plots(self):
        """Organize quantification plots into centralized structure."""
        print("\n📊 Organizing quantification plots...")
        
        quant_plots_src = self.paths["input_dirs"]["quantified_data"] / "plots"
        if not quant_plots_src.exists():
            print(f"  ⚠️ No quantification plots found at {quant_plots_src}")
            return
            
        quant_dest = self.output_dir / "03_quantification"
        
        # Look for various plot types that might be generated
        plot_types = {
            "atomic_concentration": "Elemental atomic concentration plots",
            "component_analysis": "Component-level quantification plots", 
            "validation": "Sanity check and validation plots",
            "chemistry_analysis": "Chemistry-specific component analysis"
        }
        
        # First, organize any plots directly in the plots directory
        self._copy_plots_with_manifest(quant_plots_src, quant_dest, "Quantification analysis plots")
        
        # Then check for organized subdirectories
        for plot_type, description in plot_types.items():
            src_dir = quant_plots_src / plot_type
            if src_dir.exists():
                dest_dir = quant_dest / plot_type
                dest_dir.mkdir(exist_ok=True)
                self._copy_plots_with_manifest(src_dir, dest_dir, description)
    
    def organize_correlation_plots(self):
        """Organize correlation/ML analysis plots.""" 
        print("\n📊 Organizing correlation analysis plots...")
        
        corr_plots_src = self.paths["input_dirs"]["correlator_data"]
        if not corr_plots_src.exists():
            print(f"  ⚠️ No correlation results found at {corr_plots_src}")
            return
            
        corr_dest = self.output_dir / "04_correlation_analysis"
        
        # Copy all plots from correlator results
        self._copy_plots_with_manifest(corr_plots_src, corr_dest, "ML correlation and trend analysis")
    
    def generate_summary_dashboard(self):
        """Generate high-level summary plots combining data from all steps."""
        print("\n📊 Generating summary dashboard...")
        
        summary_dest = self.output_dir / "00_summary_dashboard"
        summary_dest.mkdir(exist_ok=True)
        
        # Try to create summary plots from available data
        self._generate_workflow_overview(summary_dest)
        self._generate_data_quality_summary(summary_dest) 
    
    def _copy_plots_with_manifest(self, src_dir: Path, dest_dir: Path, description: str):
        """Copy plots and create manifest file."""
        if not src_dir.exists():
            return
            
        plot_extensions = {'.png', '.pdf', '.svg', '.jpg', '.jpeg', '.eps'}
        plot_files = []
        
        for file_path in src_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in plot_extensions:
                # Create relative path structure in destination
                rel_path = file_path.relative_to(src_dir)
                dest_path = dest_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(file_path, dest_path)
                plot_files.append({
                    "file": str(rel_path),
                    "source": str(file_path.relative_to(self.project_root)),
                    "size_mb": round(file_path.stat().st_size / 1024 / 1024, 2)
                })
        
        if plot_files:
            # Create manifest
            manifest = {
                "description": description,
                "generated": datetime.now().isoformat(),
                "source_directory": str(src_dir.relative_to(self.project_root)),
                "plot_count": len(plot_files),
                "plots": plot_files
            }
            
            manifest_path = dest_dir / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            print(f"  ✓ Organized {len(plot_files)} plots: {dest_dir.name}")
        else:
            print(f"  ⚠️ No plots found in {src_dir}")
    
    def _generate_workflow_overview(self, dest_dir: Path):
        """Generate workflow overview/status diagram showing completeness."""
        try:
            config = PLOT_CONFIG['plot_settings']
            figsize = tuple(config['figure_sizes']['overview_plot'])
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            
            # Create workflow overview with detailed steps
            steps = [
                "Raw Data\n(00_raw_data)",
                "Converted CSV\n(01_converted_csv)", 
                "Peak Fitting\n(02_fitted_results)",
                "Quantification\n(03_quantified_data)",
                "Plots\n(04_plots)",
                "Correlation\n(05_correlator_results)"
            ]
            
            step_dirs = [
                self.paths["input_dirs"]["raw_data"],
                self.project_root / "01_converted_csv",
                self.paths["input_dirs"]["fitted_data"], 
                self.paths["input_dirs"]["quantified_data"],
                self.output_dir,  # 04_plots
                self.paths["input_dirs"]["correlator_data"]
            ]
            
            y_pos = np.arange(len(steps))
            
            # Check which steps have data and count files
            data_availability = []
            file_counts = []
            for step_dir in step_dirs:
                if step_dir.exists():
                    # Count files (excluding directories and hidden files)
                    files = [f for f in step_dir.rglob('*') if f.is_file() and not f.name.startswith('.')]
                    count = len(files)
                    has_data = count > 0
                    data_availability.append(has_data)
                    file_counts.append(count)
                else:
                    data_availability.append(False)
                    file_counts.append(0)
            
            # Color scheme: green for complete, yellow for partial, red for empty
            colors = []
            for available, count in zip(data_availability, file_counts):
                if count > 50:
                    colors.append('#2ecc71')  # Bright green - complete
                elif count > 10:
                    colors.append('#f39c12')  # Orange - substantial
                elif count > 0:
                    colors.append('#e74c3c')  # Red - minimal
                else:
                    colors.append('#95a5a6')  # Gray - empty
            
            # Create horizontal bar chart
            bars = ax.barh(y_pos, [1]*len(steps), color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(steps, fontsize=config['fonts']['axis_label_size'])
            ax.set_xlim([0, 1.2])
            ax.set_xlabel('Workflow Progress', fontsize=config['fonts']['axis_label_size_large'], fontweight='bold')
            ax.set_title('XPS Workflow Status - Degree of Completeness', 
                        fontsize=config['fonts']['title_size'] + 2, fontweight='bold', pad=20)
            
            # Remove x-axis ticks
            ax.set_xticks([])
            
            # Add status annotations with file counts
            for i, (available, step, count) in enumerate(zip(data_availability, steps, file_counts)):
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
                       fontsize=config['fonts']['annotation_size'], fontweight='bold', color=color)
            
            # Add legend for color coding
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', edgecolor='black', label='Complete (>50 files)'),
                Patch(facecolor='#f39c12', edgecolor='black', label='Substantial (11-50 files)'),
                Patch(facecolor='#e74c3c', edgecolor='black', label='Minimal (1-10 files)'),
                Patch(facecolor='#95a5a6', edgecolor='black', label='Empty (0 files)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=config['fonts']['legend_size']-2,
                     framealpha=0.95, edgecolor='black')
            
            # Add timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ax.text(0.02, -0.15, f'Generated: {timestamp}', transform=ax.transAxes,
                   fontsize=config['fonts']['info_text_size'], style='italic', color='gray')
            
            # Add completion percentage
            completion_pct = sum(data_availability) / len(data_availability) * 100
            ax.text(0.98, -0.15, f'Overall Completion: {completion_pct:.0f}%', 
                   transform=ax.transAxes, fontsize=config['fonts']['annotation_size'],
                   style='italic', color='darkblue', ha='right', fontweight='bold')
            
            plt.tight_layout()
            export_config = PLOT_CONFIG['export']
            
            # Save as both workflow_overview.png and workflow_status.png
            for filename in ['workflow_overview', 'workflow_status']:
                plt.savefig(
                    dest_dir / f"{filename}.{export_config['default_format']}", 
                    dpi=config['dpi'], 
                    bbox_inches=export_config['bbox_inches'],
                    facecolor=export_config['facecolor']
                )
            plt.close()
            
            print(f"  ✓ Generated workflow status overview")
            
        except Exception as e:
            print(f"  ⚠️ Error generating workflow overview: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_data_quality_summary(self, dest_dir: Path):
        """Generate data quality summary plots."""
        try:
            # Read quantified data if available
            quant_file = self.paths["input_dirs"]["quantified_data"] / "atomic_concentration_raw.csv"
            if not quant_file.exists():
                print(f"  ⚠️ No quantification data available for quality summary")
                return
                
            df = pd.read_csv(quant_file)
            
            if df.empty:
                print(f"  ⚠️ Quantification data is empty")
                return
                
            # Create summary plots using configuration
            config = PLOT_CONFIG['plot_settings']
            figsize = tuple(config['figure_sizes']['summary_plot'])
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle('Data Quality Summary', fontsize=config['fonts']['title_size'], fontweight='bold')
            
            # Plot 1: Sample count by element
            element_cols = [col for col in df.columns if col not in ['Sample', 'Layer']]
            sample_counts = []
            for col in element_cols:
                count = df[col].notna().sum()
                sample_counts.append(count)
                
            axes[0,0].bar(element_cols, sample_counts, color='skyblue')
            axes[0,0].set_title('Data Availability by Element', fontsize=config['fonts']['subtitle_size'])
            axes[0,0].set_ylabel('Number of Samples', fontsize=config['fonts']['axis_label_size'])
            axes[0,0].tick_params(axis='x', rotation=45, labelsize=config['fonts']['tick_label_size'])
            
            # Plot 2: Average atomic percentages
            if element_cols:
                avg_values = [df[col].mean() for col in element_cols if df[col].notna().sum() > 0]
                valid_cols = [col for col in element_cols if df[col].notna().sum() > 0]
                
                if avg_values:
                    axes[0,1].pie(avg_values, labels=valid_cols, autopct='%1.1f%%', startangle=90)
                    axes[0,1].set_title('Average Elemental Composition', fontsize=config['fonts']['subtitle_size'])
            
            # Plot 3: Sample distribution
            axes[1,0].hist(df.index, bins=min(20, len(df)), color='lightgreen', alpha=0.7)
            axes[1,0].set_title('Sample Distribution', fontsize=config['fonts']['subtitle_size'])
            axes[1,0].set_xlabel('Sample Index', fontsize=config['fonts']['axis_label_size'])
            axes[1,0].set_ylabel('Frequency', fontsize=config['fonts']['axis_label_size'])
            
            # Plot 4: Data completeness heatmap
            if len(element_cols) > 0 and len(df) > 0:
                completeness = df[element_cols].notna().astype(int)
                im = axes[1,1].imshow(completeness.T, cmap='RdYlGn', aspect='auto')
                axes[1,1].set_title('Data Completeness Matrix', fontsize=config['fonts']['subtitle_size'])
                axes[1,1].set_xlabel('Sample Index', fontsize=config['fonts']['axis_label_size']) 
                axes[1,1].set_ylabel('Elements', fontsize=config['fonts']['axis_label_size'])
                axes[1,1].set_yticks(range(len(element_cols)))
                axes[1,1].set_yticklabels(element_cols)
                plt.colorbar(im, ax=axes[1,1], label='Data Available')
            
            plt.tight_layout()
            export_config = PLOT_CONFIG['export']
            plt.savefig(
                dest_dir / f"data_quality_summary.{export_config['default_format']}", 
                dpi=config['dpi'], 
                bbox_inches=export_config['bbox_inches'],
                facecolor=export_config['facecolor']
            )
            plt.close()
            
            print(f"  ✓ Generated data quality summary")
            
        except Exception as e:
            print(f"  ⚠️ Error generating data quality summary: {e}")
    
    def organize_all_plots(self):
        """Execute complete plot organization workflow."""
        print("\n" + "="*70)
        print("🎨 XPS PLOT ORGANIZATION - Professional Dual System")
        print("="*70)
        print(f"📂 Centralizing plots to: {self.output_dir}")
        print(f"📂 Source project: {self.project_root}")
        
        # Organize plots from each step
        self.organize_fitting_plots()
        self.organize_quantification_plots() 
        self.organize_correlation_plots()
        
        # Generate summary dashboard
        self.generate_summary_dashboard()
        
        # Update master index
        self.create_master_index()
        
        print("\n✅ Plot organization complete!")
        print(f"📁 All plots organized in: {self.output_dir}")
        print(f"📋 See PLOT_INDEX.md for detailed structure")
        print("="*70)


def organize_plots(project_root=None):
    """Main function to organize all XPS plots professionally."""
    organizer = XPSPlotOrganizer(project_root)
    organizer.organize_all_plots()
    return organizer.output_dir


if __name__ == "__main__":
    import sys
    project_root = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = organize_plots(project_root)
    print(f"\n🎯 Plot organization complete: {output_dir}")


# ========== LEGACY PLOTTING FUNCTIONS (maintained for compatibility) ==========

def legacy_resolve_plot_paths(config):
    """Legacy path resolution for backward compatibility."""
    root = Path(config.project_root) if hasattr(config, 'project_root') else Path.cwd()
    input_dir_raw = root / "converted_csv"
    input_dir_fits = root / "02_fitted_results"
    output_dir = root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "input_raw": input_dir_raw,
        "input_fits": input_dir_fits,
        "output_dir": output_dir,
    }


def plot_template_fit(sample_name,
                      fit_result,
                      region_name,
                      vis_settings: Dict,
                      outdir: str = "plots",
                      figsize=None,
                      dpi: int | None = None):
    """Create publication-quality plot showing template-based fit."""
    
    # Load plot configuration
    config = PLOT_CONFIG['plot_settings']
    
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
        figsize_to_use = tuple(config['figure_sizes']['single_plot'])
    else:
        figsize_to_use = tuple(figsize)
    if dpi is None:
        dpi_to_use = config['dpi']
    else:
        dpi_to_use = int(dpi)

    # Use configuration for height ratios
    height_ratios = config['height_ratios']['main_residual']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_to_use, height_ratios=height_ratios)

    x = fit_result["x"]
    # Safe getter in case fit_result is not a dict
    def _safe_get(obj, key, default=None):
        return obj.get(key, default) if isinstance(obj, dict) else default

    bg_label = f"{_safe_get(fit_result, 'background_type', 'Shirley').title()} background"

    # Main plot - NO STATISTICS HERE
    line_width = config['lines']['fit_line_width']
    fit_alpha = config['lines']['fit_alpha'] 
    comp_alpha = config['lines']['component_alpha']
    
    ax1.plot(x, fit_result["raw"], 'k-', linewidth=1.5, label="Raw data", alpha=0.8)
    ax1.plot(x, fit_result["baseline"], 'gray', linewidth=1.5, label=bg_label)
    ax1.plot(x, fit_result["corrected"], 'b-', linewidth=1.5, label="Background corrected")

    # Individual components with template names
    for name, curve in fit_result["components"].items():
        col = comp_colors.get(name, 'gray')
        ax1.fill_between(
            x,
            fit_result["baseline"],
            curve + fit_result["baseline"],
            color=col,
            alpha=comp_alpha,
            label=name,
        )

    # Total fit - use config colors and styling
    ax1.plot(
        x,
        fit_result["fit"] + fit_result["baseline"],
        color=config['colors']['fit_line'],
        linestyle='--',
        linewidth=line_width,
        alpha=fit_alpha,
        label="Total fit",
    )

    # Formatting with config font sizes
    font_config = config['fonts']
    ax1.set_ylabel("Intensity (counts)", fontsize=font_config['axis_label_size'], fontweight='bold')
    ax1.set_title(
        f"{sample_name} - {region_name} (Template: {fit_result['template_used']})",
        fontsize=font_config['title_size'],
        fontweight='bold',
        pad=config['layout']['title_pad']
    )
    
    # Legend with configuration
    legend_config = config['legend']
    ax1.legend(
        bbox_to_anchor=legend_config['bbox_anchor'], 
        loc=legend_config['position'],
        fontsize=font_config['legend_size'],
        frameon=legend_config['frameon'],
        framealpha=legend_config['alpha']
    )
    ax1.grid(True, alpha=config['lines']['grid_alpha'])
    ax1.invert_xaxis()

    # Residuals plot
    residuals = fit_result["corrected"] - fit_result["fit"]
    ax2.plot(x, residuals, color=config['colors']['residuals'], linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Binding Energy (eV)", fontsize=font_config['axis_label_size'], fontweight='bold')
    ax2.set_ylabel("Residuals", fontsize=font_config['residual_label_size'])
    ax2.grid(True, alpha=config['lines']['grid_alpha'])
    ax2.invert_xaxis()

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

    # Apply layout configuration
    layout_config = config['layout']['subplot_adjust']
    plt.subplots_adjust(
        top=layout_config['top'],
        bottom=layout_config['bottom'],
        left=layout_config['left'],
        right=layout_config['right'],
        hspace=layout_config['hspace']
    )

    # Save plot with region tag using config export settings
    export_config = PLOT_CONFIG['export']
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
      - Single or multiple layers
      - Different x-data ranges across layers (plots common region)

    Styles:
      - Corrected (data): dotted line with optional markers
      - Total Fit: solid line
    """

    if not all_layer_fits:
        raise ValueError("No layer fits provided")
        
    # Load plot configuration
    config = PLOT_CONFIG['plot_settings']

    # Style configuration (with sensible defaults)
    corrected_color = vis_settings.get('corrected_color', 'b')  # blue
    corrected_ls = vis_settings.get('corrected_linestyle', 'None')
    corrected_marker = vis_settings.get('corrected_marker', 'o')  # or None
    corrected_msize = float(vis_settings.get('corrected_markersize', 4.0))
    corrected_alpha = float(vis_settings.get('corrected_alpha', 0.9))
    corrected_markevery = vis_settings.get('corrected_markevery', None)  # int or None

    # Use config for fit styling
    fit_color = vis_settings.get('fit_color', config['colors']['fit_line'])
    fit_ls = vis_settings.get('fit_linestyle', '-')  # solid
    fit_lw = float(vis_settings.get('fit_linewidth', config['lines']['fit_line_width']))
    fit_alpha = float(vis_settings.get('fit_alpha', config['lines']['fit_alpha']))

    # Component colors (preserve your previous logic)
    layers = sorted(all_layer_fits.items(), key=lambda kv: str(kv[0]))
    n = len(layers)

    # Figure sizing and DPI from config
    if dpi is None:
        dpi_to_use = config['dpi']
    else:
        dpi_to_use = int(dpi)

    # Handle single vs multi-layer figure size using config
    if n == 1:
        figsize_default = tuple(config['figure_sizes']['single_stacked'])
        figsize_to_use = tuple(figsize) if figsize is not None else figsize_default
        fig, axes = plt.subplots(1, 1, figsize=figsize_to_use)
        axes = [axes]
    else:
        base_width, base_height = config['figure_sizes']['multi_stacked_base']
        figsize_default = (base_width, base_height * n)
        figsize_to_use = tuple(figsize) if figsize is not None else figsize_default
        fig, axes = plt.subplots(n, 1, figsize=figsize_to_use, sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = [axes]

    # Collect all unique components across all layers
    all_comps = sorted({c for _, fit in layers for c in fit["components"].keys()})
    comp_colors = vis_settings.get('component_colors', {}) or {}
    cmap_name = vis_settings.get('type', 'viridis')
    reverse = bool(vis_settings.get('reverse', False))
    missing = [c for c in all_comps if c not in comp_colors]
    if missing:
        cmap = plt.get_cmap(cmap_name)
        vals = np.linspace(0, 1, len(missing))
        if reverse:
            vals = vals[::-1]
        comp_colors.update({c: cmap(v) for c, v in zip(missing, vals)})

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
        print(f"   ⚠️  Warning: No overlapping x-range found across layers")
        print(f"      Using full range instead")
        common_x_min = min(x_mins)
        common_x_max = max(x_maxs)
    else:
        print(f"   📏 Common x-range: [{common_x_min:.2f}, {common_x_max:.2f}] eV")

    for ax, (label, fit) in zip(axes, layers):
        x = fit["x"]

        # Filter data to common x-range
        mask = (x >= common_x_min) & (x <= common_x_max)
        x_plot = x[mask]
        corrected_plot = fit["corrected"][mask]
        fit_plot = fit["fit"][mask]

        if len(x_plot) < 2:
            print(f"   ⚠️  Warning: Layer '{label}' has insufficient data in common range")
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
        font_config = config['fonts']
        ax.text(
            0.03,
            0.97,
            str(label),
            transform=ax.transAxes,
            fontsize=font_config['annotation_size'],
            fontweight='bold',
            verticalalignment='top',
            horizontalalignment='left',
        )

        ax.set_ylabel("Intensity", fontsize=font_config['axis_label_size'], fontweight='bold')
        ax.grid(alpha=config['lines']['grid_alpha'])

    # Set x-axis label using config fonts
    axes[-1].set_xlabel("Binding Energy (eV)", fontsize=font_config['axis_label_size_large'], fontweight='bold')

    # Set common x-limits for all subplots (reversed for binding energy)
    for ax in axes:
        ax.set_xlim(common_x_max, common_x_min)  # high → low
        ax.tick_params(labelsize=font_config['tick_label_size'])
        ax.grid(True, alpha=config['lines']['grid_alpha'])

    # Legend and title using config settings
    legend_config = config['legend']
    axes[0].legend(
        loc=legend_config['stacked_position'], 
        bbox_to_anchor=legend_config['stacked_bbox_anchor'], 
        fontsize=font_config['legend_size'],
        frameon=legend_config['frameon'],
        framealpha=legend_config['alpha']
    )

    # Adjust title for single vs multiple layers
    if n == 1:
        title = f"{sample_name} [{region_name}]"
    else:
        title = f"{sample_name} [{region_name}] Multilayer comparison"

    axes[0].set_title(
        title, 
        fontsize=font_config['title_size'], 
        fontweight='bold', 
        pad=config['layout']['title_pad']
    )

    # Save figure using config export and layout settings
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    export_config = PLOT_CONFIG['export']
    fn = f"{sample_name}_{region_name}_stacked.{export_config['default_format']}"
    out = outdir / fn

    # Apply layout configuration
    layout_config = config['layout']['subplot_adjust']
    fig.subplots_adjust(
        top=layout_config['top'], 
        bottom=layout_config['bottom'], 
        left=layout_config['left'], 
        right=layout_config['right'], 
        hspace=layout_config['hspace']
    )
    
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


# =====================
# Quantifier Plotting
# =====================

def plot_atomic_concentration_per_sample(
    df: pd.DataFrame,
    sample: str,
    plot_columns: List[str],
    plot_labels: List[str],
    out_dir: Path,
    cfg,
):
    sub = df[df["Sample"] == sample]
    if sub.empty:
        return None

    layers = sorted(sub["Layer"].unique().tolist())
    n_elements = len(plot_columns)
    n_layers = len(layers)

    fig, ax = plt.subplots(figsize=tuple(cfg.plot.figsize))
    indices = np.arange(n_elements)

    width = 0.8 / max(n_layers, 1)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n_layers)]

    for l_idx, layer in enumerate(layers):
        row = sub[sub["Layer"] == layer]
        if row.empty:
            vals = [np.nan] * n_elements
        else:
            row_vals = row.iloc[0]
            vals = [float(row_vals[col]) if pd.notna(row_vals[col]) else np.nan for col in plot_columns]
        ax.bar(indices + l_idx * width, vals, width=width, color=colors[l_idx], label=f"Layer {layer}")

    ax.set_xticks(indices + (n_layers - 1) * width / 2)
    ax.set_xticklabels(plot_labels, rotation=45, ha="right", fontsize=15)
    ax.set_ylabel("Atomic Concentration (%)", fontsize=15)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_title(f"{sample}: Atomic Concentration by Layer", fontsize=16, fontweight='bold')
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=2, fontsize=14, frameon=False)
    fig.tight_layout()

    fname = out_dir / f"{sanitize_filename(sample)}_atomic_concentration.png"
    fig.savefig(fname, dpi=cfg.plot.dpi)
    plt.close(fig)
    return fname


def plot_atomic_concentration_layer_comparison(
    df: pd.DataFrame,
    layer: int,
    plot_columns: List[str],
    plot_labels: List[str],
    out_dir: Path,
    cfg,
):
    sub = df[df["Layer"] == layer]
    if sub.empty:
        return None

    samples = sub["Sample"].unique().tolist()
    samples_sorted = sorted(samples)

    n_elements = len(plot_columns)
    n_samples = len(samples_sorted)
    width = 0.8 / max(n_samples, 1)

    fig, ax = plt.subplots(figsize=tuple(cfg.plot.figsize))
    indices = np.arange(n_elements)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n_samples)]

    for s_idx, sample in enumerate(samples_sorted):
        row = sub[sub["Sample"] == sample]
        if row.empty:
            vals = [np.nan] * n_elements
        else:
            row_vals = row.iloc[0]
            vals = [float(row_vals[col]) if pd.notna(row_vals[col]) else np.nan for col in plot_columns]
        ax.bar(indices + s_idx * width, vals, width=width, color=colors[s_idx], label=sample)

    ax.set_xticks(indices + (n_samples - 1) * width / 2)
    ax.set_xticklabels(plot_labels, rotation=45, ha="right", fontsize=18)
    ax.set_ylabel("Atomic Concentration (%)", fontsize=18)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_title(f"Layer {layer}: Atomic Concentration Comparison", fontsize=16, fontweight='bold')
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=2, fontsize=14, frameon=False)
    fig.tight_layout()

    fname = out_dir / f"layer_{layer}_atomic_concentration_comparison.png"
    fig.savefig(fname, dpi=cfg.plot.dpi)
    plt.close(fig)
    return fname


def generate_component_chemistry_plots(results_df: pd.DataFrame, plots_dir: Path, cfg):
    print("\nGenerating component chemistry plots...")
    plot_data = results_df[~results_df['Component_atomic_percent'].isna()].copy()
    if plot_data.empty:
        print("  ? No valid component data to plot")
        return []

    outputs = []
    # Iterate through chemistry groups from config
    for chem_name, chem_group in cfg.chemistry.groups.items():
        region_prefix = chem_group.region_prefix
        chem_data = plot_data[plot_data['Region'].str.startswith(region_prefix)].copy()
        if chem_data.empty:
            print(f"  ? No data for {chem_name} chemistry")
            continue

        # NEW: Create cross-sample component comparison plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create sample labels
        chem_data['Sample_Label'] = chem_data['Sample'].apply(
            lambda x: Path(x).stem.split('_')[-1] if '_' in x else Path(x).stem
        )
        
        # Pivot: Samples vs Components
        pivot_cross_sample = chem_data.pivot_table(
            index='Sample_Label', columns='Component', values='Component_atomic_percent', 
            aggfunc='mean', fill_value=0
        )
        
        if not pivot_cross_sample.empty:
            # Create stacked bar plot showing components across samples
            pivot_cross_sample.plot(kind='bar', ax=ax, width=0.8, 
                                   edgecolor='black', linewidth=0.5, stacked=True)
            ax.set_xlabel('Sample', fontsize=12, fontweight='bold')
            ax.set_ylabel('Component Atomic %', fontsize=12, fontweight='bold')
            ax.set_title(f'{chem_name}: Component Distribution Across Samples', 
                        fontsize=14, fontweight='bold')
            ax.legend(title='Component', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.tick_params(axis='x', rotation=45)
            
            # Add total values on top of bars
            totals = pivot_cross_sample.sum(axis=1)
            for i, (sample, total) in enumerate(totals.items()):
                if total > 0:
                    ax.text(i, total + 0.5, f'{total:.1f}%', 
                           ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plot_path = plots_dir / f'{chem_name}_cross_sample_components.png'
        plt.savefig(plot_path, dpi=cfg.plot.dpi, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_path.name}")
        plt.close()
        outputs.append(plot_path)
        
        # Also create individual component trends across samples
        create_individual_component_trends(chem_data, chem_name, plots_dir, cfg)
        
    return outputs


def create_individual_component_trends(chem_data, chem_name, plots_dir, cfg):
    """Create individual plots for each component showing trends across samples."""
    
    components = chem_data['Component'].unique()
    if len(components) == 0:
        return
    
    # Create sample labels
    chem_data['Sample_Label'] = chem_data['Sample'].apply(
        lambda x: Path(x).stem.split('_')[-1] if '_' in x else Path(x).stem
    )
    
    # Create subplots for all components
    n_components = len(components)
    n_cols = 3
    n_rows = (n_components + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, component in enumerate(components):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
            
        # Get data for this component
        comp_data = chem_data[chem_data['Component'] == component].copy()
        
        if not comp_data.empty:
            # Group by sample and average across layers if multiple
            sample_avg = comp_data.groupby('Sample_Label')['Component_atomic_percent'].mean()
            
            # Create bar plot
            sample_avg.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
            ax.set_title(f'{component}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Atomic %', fontsize=10)
            ax.set_xlabel('Sample', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            
            # Add value labels on bars
            for j, (sample, value) in enumerate(sample_avg.items()):
                if value > 0:
                    ax.text(j, value + 0.1, f'{value:.1f}', 
                           ha='center', va='bottom', fontsize=8)
        else:
            ax.set_title(f'{component} (No Data)', fontsize=12)
            ax.set_visible(False)
    
    # Hide unused subplots
    for j in range(len(components), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'{chem_name}: Individual Component Trends Across Samples', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = plots_dir / f'{chem_name}_individual_component_trends.png'
    plt.savefig(plot_path, dpi=cfg.plot.dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {plot_path.name}")


def generate_component_heatmap(results_df: pd.DataFrame, plots_dir: Path, cfg, chemistry_name: str):
    chem_group = cfg.chemistry.groups.get(chemistry_name)
    if not chem_group:
        return None

    region_prefix = chem_group.region_prefix
    plot_data = results_df[(results_df['Region'].str.startswith(region_prefix)) & (~results_df['Component_atomic_percent'].isna())].copy()
    if plot_data.empty:
        return None

    plot_data['Sample_Layer'] = plot_data['Sample'].apply(
        lambda x: Path(x).stem.split('_')[-1] if '_' in x else Path(x).stem
    ) + '_L' + plot_data['Layer'].astype(str)

    heatmap_data = plot_data.pivot_table(
        index='Component', columns='Sample_Layer', values='Component_atomic_percent', aggfunc='mean', fill_value=0
    )
    if heatmap_data.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        cmap='viridis',
        annot=False,
        fmt='.1f',
        cbar_kws={'label': 'Atomic %'},
        ax=ax,
    )
    ax.set_title(f'{chemistry_name} Component Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Sample_Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Component', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = plots_dir / f'{chemistry_name}_chemistry_heatmap.png'
    plt.savefig(plot_path, dpi=cfg.plot.dpi, bbox_inches='tight')
    plt.close(fig)
    return plot_path


# ========== DATA QUALITY PLOTTING FOR EXTRACTED REGIONS ==========

def plot_extracted_region_quality(csv_file_path, region_name, output_dirs, figsize=None, dpi=None):
    """
    Generate quality control plots for extracted XPS region data.
    
    This function creates comprehensive plots to assess data quality before fitting,
    helping users identify noisy or problematic spectra that could lead to 
    statistically meaningless fitting results.
    
    Args:
        csv_file_path (Path): Path to the extracted region CSV file (e.g., C1s_all_HR.csv)
        region_name (str): Name of the region (e.g., "C1s") 
        output_dirs (list): List of directories to save plots to 
        figsize (tuple, optional): Figure size override
        dpi (int, optional): DPI override
        
    Returns:
        list: Paths to generated plot files
    """
    import pandas as pd
    import numpy as np
    
    # Load plot configuration
    config = PLOT_CONFIG['plot_settings']
    
    try:
        # Read the extracted region data, skipping comment lines that start with #
        df = pd.read_csv(csv_file_path, comment='#')
        
        if df.empty:
            print(f"  WARNING: No data found in {csv_file_path}")
            return []
            
        # Extract binding energy and sample columns
        binding_energy = df.iloc[:, 0].values  # First column is binding energy
        sample_columns = df.columns[1:]  # Remaining columns are samples
        
        if len(sample_columns) == 0:
            print(f"  WARNING: No sample data found in {csv_file_path}")
            return []
            
        print(f"  Generating quality plots for {region_name}: {len(sample_columns)} samples, {len(binding_energy)} data points")
        
        # Use configuration for figure size and DPI
        if figsize is None:
            figsize_to_use = tuple(config['figure_sizes']['summary_plot'])
        else:
            figsize_to_use = tuple(figsize)
        if dpi is None:
            dpi_to_use = config['dpi']
        else:
            dpi_to_use = int(dpi)
            
        font_config = config['fonts']
        export_config = PLOT_CONFIG['export']
        
        # Create comprehensive quality assessment plot
        fig = plt.figure(figsize=figsize_to_use)
        
        # Create custom subplot layout
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main plot: All spectra overlaid
        ax_main = fig.add_subplot(gs[0, :])
        
        # Color map for samples
        colors = plt.cm.viridis(np.linspace(0, 1, len(sample_columns)))
        
        # Plot all samples with transparency
        for i, (sample_col, color) in enumerate(zip(sample_columns, colors)):
            intensities = df[sample_col].values
            valid_mask = ~np.isnan(intensities)
            
            if np.any(valid_mask):
                ax_main.plot(
                    binding_energy[valid_mask], 
                    intensities[valid_mask], 
                    color=color, 
                    alpha=0.7, 
                    linewidth=1.0,
                    label=sample_col if len(sample_columns) <= 10 else None  # Legend only for ≤10 samples
                )
        
        ax_main.set_xlabel("Binding Energy (eV)", fontsize=font_config['axis_label_size'], fontweight='bold')
        ax_main.set_ylabel("Intensity (counts)", fontsize=font_config['axis_label_size'], fontweight='bold') 
        ax_main.set_title(f"{region_name} Region Quality Assessment - All Samples", 
                         fontsize=font_config['title_size'], fontweight='bold')
        ax_main.grid(True, alpha=config['lines']['grid_alpha'])
        ax_main.invert_xaxis()  # XPS convention: high to low binding energy
        
        if len(sample_columns) <= 10:  # Show legend only for manageable number of samples
            ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          fontsize=font_config['legend_size'] - 4)
        
        # Statistics subplot: Intensity statistics
        ax_stats = fig.add_subplot(gs[1, 0])
        
        # Calculate statistics for each sample
        sample_stats = []
        for sample_col in sample_columns:
            intensities = df[sample_col].dropna().values
            if len(intensities) > 0:
                stats = {
                    'sample': sample_col,
                    'max': np.max(intensities),
                    'mean': np.mean(intensities),
                    'std': np.std(intensities),
                    'snr': np.max(intensities) / (np.std(intensities) + 1e-10)  # Signal-to-noise ratio
                }
                sample_stats.append(stats)
        
        if sample_stats:
            max_intensities = [s['max'] for s in sample_stats]
            snr_values = [s['snr'] for s in sample_stats]
            
            # Box plot of maximum intensities
            ax_stats.boxplot(max_intensities, patch_artist=True)
            ax_stats.set_ylabel("Max Intensity", fontsize=font_config['axis_label_size'])
            ax_stats.set_title("Intensity Distribution", fontsize=font_config['subtitle_size'])
            ax_stats.set_xlabel("All Samples", fontsize=font_config['axis_label_size'])
            ax_stats.grid(True, alpha=config['lines']['grid_alpha'])
        
        # Quality metrics subplot: Signal-to-noise analysis  
        ax_quality = fig.add_subplot(gs[1, 1])
        
        if sample_stats:
            sample_indices = range(len(sample_stats))
            ax_quality.scatter(sample_indices, snr_values, alpha=0.7, color='orange', s=50)
            ax_quality.set_xlabel("Sample Index", fontsize=font_config['axis_label_size'])
            ax_quality.set_ylabel("Signal-to-Noise Ratio", fontsize=font_config['axis_label_size'])
            ax_quality.set_title("Data Quality Metrics", fontsize=font_config['subtitle_size'])
            ax_quality.grid(True, alpha=config['lines']['grid_alpha'])
            
            # Add quality threshold line
            if len(snr_values) > 0:
                quality_threshold = np.median(snr_values) * 0.5  # 50% of median SNR as warning threshold
                ax_quality.axhline(y=quality_threshold, color='red', linestyle='--', alpha=0.7, 
                                 label=f'Quality Warning: {quality_threshold:.1f}')
                ax_quality.legend(fontsize=font_config['legend_size'] - 2)
        
        # Data coverage subplot: Missing data analysis
        ax_coverage = fig.add_subplot(gs[2, :])
        
        # Create coverage matrix
        coverage_matrix = []
        for sample_col in sample_columns:
            coverage_row = ~df[sample_col].isna()
            coverage_matrix.append(coverage_row.values)
        
        if coverage_matrix:
            coverage_matrix = np.array(coverage_matrix)
            
            # Plot coverage heatmap
            im = ax_coverage.imshow(coverage_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
            ax_coverage.set_xlabel("Data Point Index", fontsize=font_config['axis_label_size'])
            ax_coverage.set_ylabel("Samples", fontsize=font_config['axis_label_size'])
            ax_coverage.set_title("Data Coverage Matrix (Green=Valid, Red=Missing)", 
                                fontsize=font_config['subtitle_size'])
            
            # Set y-tick labels to sample names (if reasonable number)
            if len(sample_columns) <= 20:
                ax_coverage.set_yticks(range(len(sample_columns)))
                ax_coverage.set_yticklabels([str(col)[:10] for col in sample_columns], 
                                          fontsize=font_config['tick_label_size'] - 2)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_coverage, shrink=0.8)
            cbar.set_label('Data Available', fontsize=font_config['axis_label_size'])
        
        # Add summary text box
        if sample_stats:
            n_samples = len(sample_stats)
            avg_max = np.mean(max_intensities)
            avg_snr = np.mean(snr_values)
            low_quality_count = sum(1 for snr in snr_values if snr < np.median(snr_values) * 0.5)
            
            summary_text = (f"Quality Summary:\n"
                          f"Samples: {n_samples}\n"
                          f"Avg Max Intensity: {avg_max:.0f}\n"
                          f"Avg SNR: {avg_snr:.1f}\n"
                          f"Low Quality: {low_quality_count}/{n_samples}")
            
            fig.text(0.02, 0.02, summary_text, fontsize=font_config['info_text_size'], 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    verticalalignment='bottom')
        
        # Apply layout configuration 
        plt.tight_layout()
        
        # Save plots to all specified directories
        saved_paths = []
        filename = f"{region_name}_quality_assessment.{export_config['default_format']}"
        
        for output_dir in output_dirs:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            plot_path = output_path / filename
            fig.savefig(
                plot_path,
                dpi=dpi_to_use,
                bbox_inches=export_config['bbox_inches'],
                facecolor=export_config['facecolor'],
                transparent=export_config['transparent']
            )
            saved_paths.append(plot_path)
            
        plt.close(fig)
        
        print(f"    Quality plots saved: {len(saved_paths)} locations")
        return saved_paths
        
    except Exception as e:
        print(f"  ERROR: Error generating quality plots for {region_name}: {e}")
        return []


def plot_all_extracted_regions_quality(converted_csv_dir, plots_dir=None, project_root=None):
    """
    Generate quality control plots for all extracted XPS regions.
    
    Args:
        converted_csv_dir (Path): Directory containing extracted CSV files (01_converted_csv)
        plots_dir (Path, optional): Main plots directory (04_plots)
        project_root (Path, optional): Project root directory
        
    Returns:
        dict: Region name -> list of plot paths
    """
    
    converted_csv_dir = Path(converted_csv_dir)
    
    if not converted_csv_dir.exists():
        print(f"ERROR: Converted CSV directory not found: {converted_csv_dir}")
        return {}
    
    # Setup output directories
    if plots_dir is None:
        if project_root:
            plots_dir = Path(project_root) / "04_plots"
        else:
            plots_dir = converted_csv_dir.parent / "04_plots"
    
    plots_dir = Path(plots_dir)
    data_conversion_plots_dir = plots_dir / "01_data_conversion"
    
    # Create output directories
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_conversion_plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("XPS DATA QUALITY ASSESSMENT - Extracted Regions")
    print("="*70)
    print(f"Source: {converted_csv_dir}")
    print(f"Plots: {data_conversion_plots_dir}")
    
    all_plots = {}
    processed_regions = 0
    
    # Process each region directory
    for region_dir in converted_csv_dir.iterdir():
        if not region_dir.is_dir():
            continue
            
        region_name = region_dir.name
        
        # Look for aggregated CSV files (e.g., C1s_all_HR.csv, F1s_all_HR.csv)
        hr_files = list(region_dir.glob("*_all_HR.csv"))
        survey_files = list(region_dir.glob("*_all_survey.csv"))
        
        region_plots = []
        
        # Process HR files (high resolution - preferred for fitting)
        for csv_file in hr_files:
            print(f"\nProcessing {region_name} - High Resolution")
            
            # Define output directories for dual saving
            output_dirs = [
                region_dir,  # Save in source region directory  
                data_conversion_plots_dir  # Save in centralized plots directory
            ]
            
            plots = plot_extracted_region_quality(
                csv_file_path=csv_file,
                region_name=f"{region_name}_HR",
                output_dirs=output_dirs
            )
            region_plots.extend(plots)
            
        # Process survey files (if no HR available)
        if not hr_files and survey_files:
            for csv_file in survey_files:
                print(f"\nProcessing {region_name} - Survey")
                
                output_dirs = [
                    region_dir,
                    data_conversion_plots_dir
                ]
                
                plots = plot_extracted_region_quality(
                    csv_file_path=csv_file, 
                    region_name=f"{region_name}_survey",
                    output_dirs=output_dirs
                )
                region_plots.extend(plots)
        
        if region_plots:
            all_plots[region_name] = region_plots
            processed_regions += 1
        else:
            print(f"  WARNING: No suitable CSV files found for {region_name}")
    
    print(f"\nQuality assessment complete!")
    print(f"Processed regions: {processed_regions}")
    print(f"Total plots generated: {sum(len(plots) for plots in all_plots.values())}")
    print(f"Centralized plots: {data_conversion_plots_dir}")
    print("="*70)
    
    return all_plots

