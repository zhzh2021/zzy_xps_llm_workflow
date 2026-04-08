"""
Wrapper for the unified XPS workflow with automatic triage and routing.

This tool provides a complete production-ready XPS analysis pipeline with:

1. **Triage & Quality Gate (Mandatory)**:
   - Auto-detects file types (.spe, .vgd, .csv, .xy, etc.)
   - Validates data integrity (SNR, energy axis, spatial parameters)
   - Routes to appropriate workflow

2. **Standard Workflow** (for spectra):
   - XPS_Reader: Convert raw files to CSV
   - XPS_Fitter: Peak fitting with templates
   - XPS_Quantifier: Elemental composition
   - XPS_Plotter: Visualizations

3. **Map Workflow** (for 2D/3D maps):
   - XPS_Mapper: PCA clustering, MCR analysis
   - XPS_Plotter: Spatial maps and components

4. **Mixed Datasets**: Automatically runs both workflows sequentially

Usage:
    run(project_root="path/to/project")
    
The workflow scans 00_raw_data/ folder, triages all files, and executes
the appropriate pipeline(s) without user intervention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .tool_runner import run_python_script


SCRIPT_PATH = Path(__file__).resolve().parent / "real_xps_workflow.py"


def run(project_root: Optional[str] = None, extra_args: Optional[Sequence[str]] = None):
    """
    Run the unified XPS workflow with automatic triage and routing.
    
    Args:
        project_root: Path to project directory (default: current directory)
        extra_args: Additional command-line arguments
        
    Returns:
        Execution result from run_python_script
        
    The workflow automatically:
    - Scans 00_raw_data/ for all XPS files
    - Triages each file to determine type (standard vs map)
    - Validates data quality
    - Routes to appropriate workflow
    - Handles mixed datasets (runs both workflows)
    """
    args = list(extra_args or [])
    if project_root:
        args.insert(0, str(project_root))
    return run_python_script(
        SCRIPT_PATH,
        project_root=project_root,
        extra_args=args,
        friendly_name="Unified XPS Workflow (Triage + Standard/Map)",
    )


__all__ = ["run"]
