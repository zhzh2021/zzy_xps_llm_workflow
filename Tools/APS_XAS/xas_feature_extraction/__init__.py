"""
XAS Feature Extraction Module

Extracts comprehensive features from normalized XAS spectra for ML analysis.
Outputs to: project_root/xas_results/02_analyzed_data/extracted_features/
"""

from .xas_feature_extractor import XASFeatureExtractor

__all__ = ['XASFeatureExtractor']
