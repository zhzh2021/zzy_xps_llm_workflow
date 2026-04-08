"""
XPS Workflow Manager - Centralized Path and Configuration Management

This module ensures seamless data flow between all XPS analysis modules:
1. XPS_Reader    (Raw data → CSV)
2. XPS_Fitter    (CSV → Fitted results)  
3. XPS_Quantifier (Fitted results → Quantified data)
4. XPS_Plotter   (Any step → Plots)
5. XPS_Correlator (Quantified data → ML analysis)

The workflow creates a standardized directory structure where each step's 
output becomes the input for the next step.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List
import os

# Handle optional yaml dependency
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("⚠️  PyYAML not available. Configuration file loading will be limited.")


@dataclass
class XPSWorkflowConfig:
    """Centralized configuration for the entire XPS analysis workflow."""

    # Project root - main directory for all XPS work
    project_root: Path

    # Data directories in processing order
    # Step 0: User drops raw files (.spe, .vgd, etc.)
    raw_data_dir: Path
    csv_output_dir: Path        # Step 1: XPS_Reader output → XPS_Fitter input
    fits_output_dir: Path       # Step 2: XPS_Fitter output → XPS_Quantifier input
    quant_output_dir: Path      # Step 3: XPS_Quantifier output → XPS_Correlator input
    plots_output_dir: Path      # Plots: XPS_Plotter outputs
    correlator_output_dir: Path  # Step 4: XPS_Correlator output

    # Configuration directories
    config_dir: Path
    template_dir: Path

    # Working directories
    temp_dir: Path
    logs_dir: Path

    def __post_init__(self):
        """Ensure all directories exist after initialization."""
        self.create_directories()

    def create_directories(self):
        """Create all necessary directories for the workflow."""
        directories = [
            self.raw_data_dir,
            self.csv_output_dir,
            self.fits_output_dir,
            self.quant_output_dir,
            self.plots_output_dir,
            self.correlator_output_dir,
            self.config_dir,
            self.template_dir,
            self.temp_dir,
            self.logs_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> 'XPSWorkflowConfig':
        """
        Create workflow config from project root directory.

        Args:
            project_root: Main project directory path

        Returns:
            XPSWorkflowConfig: Configured workflow manager
        """
        project_root = Path(project_root).resolve()

        # If user passed a file or a known subfolder (e.g., 00_raw_data), move up to the project root
        if project_root.is_file():
            project_root = project_root.parent

        known_subdirs = {
            "00_raw_data",
            "01_converted_csv",
            "02_fitted_results",
            "03_quantified_data",
            "04_plots",
            "06_correlator_results",
            "xps_config",
            "_temp",
            "_logs",
        }
        if project_root.name in known_subdirs:
            project_root = project_root.parent

        # Check if we're in the main repo directory and need to go to project_root subdirectory
        actual_project_root = project_root
        if (project_root / "zzy_llm" / "project_root").exists():
            actual_project_root = project_root / "zzy_llm" / "project_root"

        return cls(
            project_root=actual_project_root,
            raw_data_dir=actual_project_root / "00_raw_data",
            csv_output_dir=actual_project_root / "01_converted_csv",
            fits_output_dir=actual_project_root / "02_fitted_results",
            quant_output_dir=actual_project_root / "03_quantified_data",
            plots_output_dir=actual_project_root / "04_plots",
            correlator_output_dir=actual_project_root / "06_correlator_results",
            config_dir=actual_project_root / "xps_config",
            template_dir=actual_project_root / "xps_config" / "LIB_fit_template",
            temp_dir=actual_project_root / "_temp",
            logs_dir=actual_project_root / "_logs"
        )

    def get_workflow_directories(self) -> Dict[str, Path]:
        """
        Get workflow directories for demo and display purposes.

        Returns:
            Dictionary mapping directory names to their paths
        """
        return {
            'raw_data': self.raw_data_dir,
            'converted_csv': self.csv_output_dir,
            'fitted_results': self.fits_output_dir,
            'quantified_data': self.quant_output_dir,
            'plots': self.plots_output_dir,
            'correlator_results': self.correlator_output_dir,
            'config': self.config_dir,
            'templates': self.template_dir,
        }

    def get_workflow_paths(self) -> Dict[str, Path]:
        """
        Get all workflow paths as a dictionary for easy module integration.

        Returns:
            Dictionary mapping step names to their respective directories
        """
        return {
            # Input/Output for each step
            'raw_input': self.raw_data_dir,
            'reader_output': self.csv_output_dir,
            'fitter_input': self.csv_output_dir,
            'fitter_output': self.fits_output_dir,
            'quantifier_input': self.fits_output_dir,
            'quantifier_output': self.quant_output_dir,
            'correlator_input': self.quant_output_dir,
            'correlator_output': self.correlator_output_dir,
            'plotter_output': self.plots_output_dir,

            # Configuration
            'config': self.config_dir,
            'templates': self.template_dir,
            'project_config': self.config_dir / "project_setting.yaml",

            # Working directories
            'temp': self.temp_dir,
            'logs': self.logs_dir,
            'project_root': self.project_root
        }

    def print_workflow_structure(self):
        """Print the complete workflow directory structure."""
        print("\n" + "="*70)
        print("🏗️  XPS WORKFLOW DIRECTORY STRUCTURE")
        print("="*70)
        print(f"📂 Project Root: {self.project_root}")
        print()

        workflow_steps = [
            ("00_raw_data", "Raw XPS files (.spe, .vgd, etc.)", "📥"),
            ("01_converted_csv", "XPS_Reader output → XPS_Fitter input", "📊"),
            ("02_fitted_results", "XPS_Fitter output → XPS_Quantifier input", "📈"),
            ("03_quantified_data", "XPS_Quantifier output → XPS_Correlator input", "🔢"),
            ("04_plots", "XPS_Plotter output (all steps)", "📊"),
            ("06_correlator_results", "XPS_Correlator output (ML analysis)", "🤖"),
            ("xps_config", "Configuration files and templates", "⚙️"),
            ("_temp", "Temporary working files", "🗂️"),
            ("_logs", "Processing logs", "📝")
        ]

        for folder, description, emoji in workflow_steps:
            path = self.project_root / folder
            status = "✅" if path.exists() else "❌"
            print(f"{status} {emoji} {folder:<25} {description}")

        print()
        print("📋 Workflow Steps:")
        print("  1️⃣  Drop raw XPS files in '00_raw_data'")
        print("  2️⃣  Run XPS_Reader → creates CSV files in '01_converted_csv'")
        print("  3️⃣  Run XPS_Fitter → creates fitted results in '02_fitted_results'")
        print("  4️⃣  Run XPS_Quantifier → creates quantified data in '03_quantified_data'")
        print("  5️⃣  Run XPS_Correlator → creates ML analysis in '05_correlator_results'")
        print("  📊 Run XPS_Plotter at any step → creates plots in '04_plots'")
        print("="*70 + "\n")


def resolve_project_root_from_env_or_yaml() -> Optional[Path]:
    """
    Resolve project root from environment variable or YAML configuration.

    Returns:
        Path to project root or None if not found
    """
    # Check environment variable first
    env_root = os.getenv('XPS_PROJECT_ROOT')
    if env_root:
        return Path(env_root).resolve()

    # Try to find YAML config in current working directory tree
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        # Check both direct xps_config and project_root/xps_config
        config_locations = [
            parent / "xps_config" / "project_setting.yaml",
            parent / "project_root" / "xps_config" / "project_setting.yaml"
        ]
        
        for config_file in config_locations:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                        yaml_root = config.get(
                            'project_info', {}).get('project_root')
                        if yaml_root:
                            return Path(yaml_root).resolve()
                        # If no custom root in YAML, use project_root if it's in the path
                        if "project_root" in config_file.parts:
                            return config_file.parent.parent.resolve()
                        return parent.resolve()
                except Exception:
                    continue

    return None


def get_workflow_config(project_root: Optional[str | Path] = None) -> XPSWorkflowConfig:
    """
    Get workflow configuration, auto-detecting project root if not provided.

    Args:
        project_root: Optional project root path. If None, attempts auto-detection.

    Returns:
        XPSWorkflowConfig: Workflow configuration

    Raises:
        ValueError: If project root cannot be determined
    """
    if project_root is None:
        # Try to auto-detect from environment or YAML
        detected_root = resolve_project_root_from_env_or_yaml()
        if detected_root is None:
            # Fall back to current working directory
            project_root = Path.cwd()
            print(
                f"[WARNING] Using current directory as project root: {project_root}")
        else:
            project_root = detected_root
            print(f"[INFO] Auto-detected project root: {project_root}")

    return XPSWorkflowConfig.from_project_root(project_root)


def update_module_paths(module_name: str, workflow_config: XPSWorkflowConfig) -> Dict[str, Path]:
    """
    Get the specific input/output paths for a given XPS module.

    Args:
        module_name: Name of the XPS module ('reader', 'fitter', 'quantifier', etc.)
        workflow_config: Workflow configuration

    Returns:
        Dictionary with input_dir, output_dir, config_dir paths for the module
    """
    paths = workflow_config.get_workflow_paths()

    module_configs = {
        'reader': {
            'input_dir': paths['raw_input'],
            'output_dir': paths['reader_output'],
            'config_dir': paths['config'],
            'description': 'Convert raw XPS files to CSV format'
        },
        'fitter': {
            'input_dir': paths['fitter_input'],
            'output_dir': paths['fitter_output'],
            'config_dir': paths['config'],
            'template_dir': paths['templates'],
            'description': 'Fit XPS spectra with peak models'
        },
        'quantifier': {
            'input_dir': paths['quantifier_input'],
            'output_dir': paths['quantifier_output'],
            'config_dir': paths['config'],
            'description': 'Quantify atomic concentrations'
        },
        'plotter': {
            'input_dirs': [paths['reader_output'], paths['fitter_output'], paths['quantifier_output']],
            'output_dir': paths['plotter_output'],
            'config_dir': paths['config'],
            'description': 'Create plots and visualizations'
        },
        'correlator': {
            'input_dir': paths['correlator_input'],
            'output_dir': paths['correlator_output'],
            'config_dir': paths['config'],
            'description': 'ML-based correlation analysis'
        }
    }

    if module_name not in module_configs:
        raise ValueError(
            f"Unknown module: {module_name}. Available: {list(module_configs.keys())}")

    return module_configs[module_name]


# Convenience function for backward compatibility
def resolve_reader_paths(project_root: Optional[str | Path] = None) -> Dict[str, str]:
    """
    Legacy function for XPS_Reader compatibility.

    Args:
        project_root: Project root path

    Returns:
        Dictionary with raw_data_dir, output_dir, config_file paths
    """
    config = get_workflow_config(project_root)
    paths = config.get_workflow_paths()

    return {
        "raw_data_dir": str(paths['raw_input']),
        "output_dir": str(paths['reader_output']),
        "config_file": str(paths['project_config'])
    }


def create_workflow_directory_structure(config: XPSWorkflowConfig) -> None:
    """
    Create the complete directory structure for the XPS workflow.

    This function ensures all necessary directories exist for the unified workflow.
    """
    config.create_directories()
    print(
        f"✓ Created unified XPS workflow directory structure at: {config.project_root}")


# =============================================================================
# CLI Interface for Testing and Setup
# =============================================================================

if __name__ == "__main__":
    # Demo/test the workflow manager
    import sys

    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = input(
            "Enter project root path (or press Enter for auto-detection): ").strip()
        if not project_root:
            project_root = None

    try:
        config = get_workflow_config(project_root)
        config.print_workflow_structure()

        # Show module configurations
        print("\n📋 Module Configurations:")
        for module in ['reader', 'fitter', 'quantifier', 'plotter', 'correlator']:
            module_config = update_module_paths(module, config)
            print(f"\n🔧 {module.upper()}:")
            print(f"   Description: {module_config['description']}")
            if 'input_dir' in module_config:
                print(f"   Input:  {module_config['input_dir']}")
            if 'input_dirs' in module_config:
                print(f"   Inputs: {module_config['input_dirs']}")
            print(f"   Output: {module_config['output_dir']}")

    except Exception as e:
        print(f"❌ Error: {e}")
