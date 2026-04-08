"""
APS XAS Analysis Tools

A modular XAS (X-ray Absorption Spectroscopy) data analysis package
designed for integration with the zzy_llm agent framework.

Modules:
- xas_reader: Data loading and file format support
- xas_analyzer: Core XAS processing with Larch
- xas_plotter: Visualization and plotting tools
- xas_workflow: Main orchestrator for complete analysis pipelines

Supported file formats:
- XDI (standard beamline format)
- ASCII text files (energy, I0, It columns)
- Direct μ(E) data files

Analysis capabilities:
- XANES normalization and pre-edge subtraction
- EXAFS background removal and Fourier transforms
- Feature extraction for machine learning
- Publication-quality plotting
- Batch processing workflows
"""

from .xas_reader.xas_reader import load_xas_file, load_xas_batch, read_xas_file
from .xas_reader.xas_reference_loader import load_xas_reference
from .xas_analyzer.xas_analyzer_main import XASProcessor, create_summary_table
from .xas_plotter.xas_plotter_main import XASPlotter
from .xas_workflow import XASAutomatedProcessor, run_xas_workflow, analyze_single_xas_file, run_xas_automated_workflow

__version__ = "0.1.0"
__author__ = "zzy_llm framework"

__all__ = [
    # Reader functions
    'load_xas_file',
    'load_xas_batch',
    'read_xas_file',
    'load_xas_reference',

    # Analyzer classes/functions
    'XASProcessor',
    'create_summary_table',

    # Plotter classes
    'XASPlotter',

    # Workflow classes/functions
    'XASAutomatedProcessor',
    'run_xas_workflow',
    'run_xas_automated_workflow',
    'analyze_single_xas_file',
]