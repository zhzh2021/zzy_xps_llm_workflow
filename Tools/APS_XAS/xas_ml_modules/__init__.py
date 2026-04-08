"""
XAS Machine Learning Modules

Modular, plug-and-play tools for machine learning analysis of XAS datasets.

Modules:
- xas_feature_extractor: Extract interpretable features from XAS spectra
- xas_batch_assembler: Assemble individual results into ML-ready datasets
- xas_pca_analyzer: Principal Component Analysis for dimensionality reduction (feature-based)
- xas_spectrum_pca: Whole-spectrum PCA (discovers structure without predefined features)
- xas_experiment_planner: Use PCA to guide experimental design (NEW!)
- xas_clustering: Clustering algorithms with validation
- xas_trend_analyzer: Correlation analysis and outlier detection
- xas_batch_processor: Orchestrator for full ML pipeline

All modules follow a consistent design pattern:
- Configuration-driven (load from xas_config/xas_ml_settings.yaml)
- Agent-friendly (simple APIs, structured outputs)
- Logging everywhere (easy debugging)
- Unit testable (isolated functionality)

See xas_ml_module_design_guide.md for development guidelines.
"""

from pathlib import Path

__version__ = "0.1.0"
__author__ = "XAS Workflow Team"

# Version info
MODULE_DIR = Path(__file__).parent
CONFIG_DIR = MODULE_DIR.parent / "xas_config"

# Configuration utilities
from .config_utils import ConfigLoader

# Feature extraction (from xas_feature_extraction)
try:
    from ..xas_feature_extraction.xas_feature_extractor import (
        XASFeatureExtractor,
        extract_features_from_sample,
        extract_features_from_batch
    )
except (ImportError, ValueError):
    # Fallback for when running as script
    import sys
    sys.path.insert(0, str(MODULE_DIR.parent))
    from xas_feature_extraction.xas_feature_extractor import (
        XASFeatureExtractor,
        extract_features_from_sample,
        extract_features_from_batch
    )

# Batch assembly
from .xas_batch_assembler import (
    XASBatchAssembler,
    assemble_dataset_from_results,
    save_dataset_to_json,
    load_dataset_from_json
)

# PCA analysis
from .xas_pca_analyzer import (
    XASPCAAnalyzer,
    perform_pca_analysis,
    get_top_features_per_component
)

# Whole-spectrum PCA (new!)
from .xas_spectrum_pca import (
    XASSpectrumPCA,
    SpectrumPCAResult
)

# Experiment planner (uses PCA for experiment design)
from .xas_experiment_planner import (
    XASExperimentPlanner,
    PCInterpretation,
    ExperimentSuggestion
)

# Clustering
from .xas_clusterer import (
    XASClusterer,
    perform_clustering,
    get_cluster_members
)

# Trend analysis
from .xas_trend_analyzer import (
    XASTrendAnalyzer,
    perform_trend_analysis,
    get_top_correlations
)

__all__ = [
    'ConfigLoader',
    'XASFeatureExtractor',
    'extract_features_from_sample',
    'extract_features_from_batch',
    'XASBatchAssembler',
    'assemble_dataset_from_results',
    'save_dataset_to_json',
    'load_dataset_from_json',
    'XASPCAAnalyzer',
    'perform_pca_analysis',
    'get_top_features_per_component',
    'XASSpectrumPCA',
    'SpectrumPCAResult',
    'XASExperimentPlanner',
    'PCInterpretation',
    'ExperimentSuggestion',
    'XASClusterer',
    'perform_clustering',
    'get_cluster_members',
    'XASTrendAnalyzer',
    'perform_trend_analysis',
    'get_top_correlations',
]
DEFAULT_CONFIG_FILE = CONFIG_DIR / "xas_ml_settings.yaml"
<<<<<<< Updated upstream

# Import main classes when available
# (Will be uncommented as modules are implemented)

# from .xas_feature_extractor import XASFeatureExtractor
# from .xas_batch_assembler import XASBatchAssembler
# from .xas_pca_analyzer import XASPCAAnalyzer
# from .xas_clustering import XASClusterer
# from .xas_trend_analyzer import XASTrendAnalyzer
# from .xas_batch_processor import XASBatchProcessor

# Keep __all__ defined above; do not overwrite.
=======
>>>>>>> Stashed changes
