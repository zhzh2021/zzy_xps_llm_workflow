"""
XAS Batch Assembler Module

Assembles individual XAS sample results into ML-ready datasets.
Extracts features, creates feature matrices, aggregates metadata.

This module bridges single-sample processing and batch ML analysis.

Author: XAS ML Integration Team
Date: 2026-03-03
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime

# Try to import pandas (optional)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Metadata handling will be limited.")

# Local imports
try:
    from xas_analyzer.xas_models import (
        XASSampleResult, 
        XASFeatures, 
        XASDataset
    )
    from xas_feature_extraction.xas_feature_extractor import XASFeatureExtractor
    from xas_ml_modules.config_utils import ConfigLoader
except ImportError:
    from ..xas_analyzer.xas_models import (
        XASSampleResult, 
        XASFeatures, 
        XASDataset
    )
    from ..xas_feature_extraction.xas_feature_extractor import XASFeatureExtractor
    from .config_utils import ConfigLoader


logger = logging.getLogger(__name__)


class XASBatchAssembler:
    """
    Assemble individual XAS results into batch datasets for ML analysis.
    
    This class:
    1. Extracts features from each sample using XASFeatureExtractor
    2. Creates feature matrix (samples × features)
    3. Aggregates metadata and quality flags
    4. Applies quality filters
    5. Returns XASDataset ready for PCA/clustering
    
    Usage:
        assembler = XASBatchAssembler()
        dataset = assembler.assemble_dataset(sample_results)
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize batch assembler.
        
        Args:
            config_path: Path to YAML config (optional, auto-detected if None)
        """
        self.config = ConfigLoader(config_path)
        self.batch_config = self.config.get_section('batch_processing')
        
        # Initialize feature extractor
        self.feature_extractor = XASFeatureExtractor(config_path)
        
        logger.info("XASBatchAssembler initialized")
    
    def assemble_dataset(
        self,
        sample_results: List[XASSampleResult],
        metadata: Optional[Dict[str, List[Any]]] = None,
        dataset_id: Optional[str] = None
    ) -> XASDataset:
        """
        Assemble XAS samples into a complete dataset.
        
        Main entry point for batch assembly.
        
        Args:
            sample_results: List of XASSampleResult objects
            metadata: Optional metadata dictionary {column: [values]}
            dataset_id: Optional identifier for the dataset
            
        Returns:
            XASDataset with feature matrix and metadata
            
        Raises:
            ValueError: If no valid samples after quality filtering
        """
        if len(sample_results) == 0:
            raise ValueError("Cannot assemble empty dataset")
        
        logger.info(f"Assembling dataset from {len(sample_results)} samples")
        
        # Step 1: Extract features from all samples
        features_list = self._extract_features_batch(sample_results)
        
        # Step 2: Apply quality filtering
        filtered_results, filtered_features = self._apply_quality_filter(
            sample_results, features_list
        )
        
        if len(filtered_results) == 0:
            raise ValueError("No samples passed quality filters")
        
        logger.info(f"Quality filtering: {len(filtered_results)}/{len(sample_results)} samples passed")
        
        # Step 3: Create feature matrix
        feature_matrix, feature_names = self._create_feature_matrix(filtered_features)
        
        # Step 4: Aggregate metadata
        sample_names = [result.sample_name for result in filtered_results]
        metadata_dict = self._aggregate_metadata(filtered_results, metadata)
        
        # Step 5: Collect quality flags
        quality_flags = self._collect_quality_flags(filtered_results)
        
        # Step 6: Create XASDataset
        dataset = XASDataset(
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            sample_names=sample_names,
            metadata_dict=metadata_dict,
            quality_flags=quality_flags,
            n_samples=len(filtered_results),
            n_features=len(feature_names),
            dataset_id=dataset_id or f"xas_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            creation_timestamp=datetime.now()
        )
        
        logger.info(f"Dataset assembled: {dataset.n_samples} samples × {dataset.n_features} features")
        return dataset
    
    def _extract_features_batch(
        self, 
        sample_results: List[XASSampleResult]
    ) -> List[XASFeatures]:
        """
        Extract features from all samples.
        
        Args:
            sample_results: List of XASSampleResult objects
            
        Returns:
            List of XASFeatures objects
        """
        logger.info("Extracting features from all samples...")
        features_list = self.feature_extractor.extract_features_batch(sample_results)
        return features_list
    
    def _apply_quality_filter(
        self,
        sample_results: List[XASSampleResult],
        features_list: List[XASFeatures]
    ) -> Tuple[List[XASSampleResult], List[XASFeatures]]:
        """
        Apply quality filters to samples.
        
        Filters based on:
        - Quality scores (if available)
        - Feature validity (no NaN/inf)
        - User-defined thresholds from config
        
        Args:
            sample_results: List of XASSampleResult objects
            features_list: List of XASFeatures objects
            
        Returns:
            Tuple of (filtered_results, filtered_features)
        """
        filter_config = self.batch_config.get('quality_filter', {})
        
        # Handle if quality_filter is a string (from YAML)
        if isinstance(filter_config, str):
            # Convert string filter to simple enabled flag
            filter_enabled = filter_config != 'all'
            min_quality_score = 0.0
        elif isinstance(filter_config, dict):
            filter_enabled = filter_config.get('enabled', True)
            min_quality_score = filter_config.get('min_quality_score', 0.0)
        else:
            filter_enabled = True
            min_quality_score = 0.0
        
        if not filter_enabled:
            logger.info("Quality filtering disabled")
            return sample_results, features_list
        
        filtered_results = []
        filtered_features = []
        
        for result, features in zip(sample_results, features_list):
            # Check if features are valid (not all None/NaN)
            if not self._is_feature_valid(features):
                logger.warning(f"Sample {result.sample_name} has invalid features, excluding")
                continue
            
            # Check quality score if available
            if hasattr(result, 'quality_score') and result.quality_score is not None:
                if result.quality_score < min_quality_score:
                    logger.warning(
                        f"Sample {result.sample_name} quality score "
                        f"{result.quality_score:.2f} < {min_quality_score}, excluding"
                    )
                    continue
            
            # Sample passed all filters
            filtered_results.append(result)
            filtered_features.append(features)
        
        return filtered_results, filtered_features
    
    def _is_feature_valid(self, features: XASFeatures) -> bool:
        """Check if features object contains valid numeric data."""
        feature_dict = features.model_dump()

        numeric_values = []
        for key, value in feature_dict.items():
            if value is None:
                continue
            if isinstance(value, (list, dict)):
                continue
            try:
                v = float(value)
            except (ValueError, TypeError):
                continue
            if np.isnan(v) or np.isinf(v):
                continue
            numeric_values.append(v)

        if len(numeric_values) == 0:
            return False

        valid_count = len(numeric_values)
        min_valid = max(1, int(len(feature_dict) * 0.5))
        return valid_count >= min_valid

    def _create_feature_matrix(
        self,
        features_list: List[XASFeatures]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature matrix from list of XASFeatures.
        
        Args:
            features_list: List of XASFeatures objects
            
        Returns:
            Tuple of (feature_matrix, feature_names)
            feature_matrix has shape (n_samples, n_features)
        """
        if len(features_list) == 0:
            return np.array([]), []
        
        # Get feature names from first sample
        first_features = features_list[0].model_dump()
        feature_names = list(first_features.keys())
        
        # Create matrix
        n_samples = len(features_list)
        n_features = len(feature_names)
        feature_matrix = np.zeros((n_samples, n_features))
        
        for i, features in enumerate(features_list):
            feature_dict = features.model_dump()
            for j, name in enumerate(feature_names):
                value = feature_dict[name]
                # Handle None, NaN, inf
                if value is None or np.isnan(value) or np.isinf(value):
                    feature_matrix[i, j] = np.nan
                else:
                    feature_matrix[i, j] = float(value)
        
        logger.info(f"Feature matrix created: {feature_matrix.shape}")
        return feature_matrix, feature_names
    
    def _aggregate_metadata(
        self,
        sample_results: List[XASSampleResult],
        user_metadata: Optional[Dict[str, List[Any]]] = None
    ) -> Optional[Dict[str, List[Any]]]:
        """
        Aggregate metadata from samples and user-provided data.
        
        Args:
            sample_results: List of XASSampleResult objects
            user_metadata: Optional user-provided metadata
            
        Returns:
            Dictionary of metadata lists, or None if no metadata
        """
        metadata_dict = {}
        
        # Extract metadata from sample results
        for i, result in enumerate(sample_results):
            # Sample name (always include)
            if 'sample_name' not in metadata_dict:
                metadata_dict['sample_name'] = []
            metadata_dict['sample_name'].append(result.sample_name)
            
            # Quality score (if available)
            if hasattr(result, 'quality_score') and result.quality_score is not None:
                if 'quality_score' not in metadata_dict:
                    metadata_dict['quality_score'] = []
                metadata_dict['quality_score'].append(result.quality_score)
            
            # Edge energy (if available)
            if hasattr(result, 'e0') and result.e0 is not None:
                if 'e0' not in metadata_dict:
                    metadata_dict['e0'] = []
                metadata_dict['e0'].append(result.e0)
            
            # Processing timestamp (if available)
            if hasattr(result, 'processing_timestamp'):
                if 'processing_timestamp' not in metadata_dict:
                    metadata_dict['processing_timestamp'] = []
                metadata_dict['processing_timestamp'].append(result.processing_timestamp)
        
        # Merge with user-provided metadata
        if user_metadata is not None:
            for key, values in user_metadata.items():
                if len(values) == len(sample_results):
                    metadata_dict[key] = values
                else:
                    logger.warning(
                        f"User metadata '{key}' has {len(values)} values "
                        f"but {len(sample_results)} samples, skipping"
                    )
        
        return metadata_dict if metadata_dict else None
    
    def _collect_quality_flags(
        self,
        sample_results: List[XASSampleResult]
    ) -> Dict[str, List[str]]:
        """
        Collect quality flags from all samples.
        
        Args:
            sample_results: List of XASSampleResult objects
            
        Returns:
            Dictionary mapping sample names to quality flags
        """
        quality_flags = {}
        
        for result in sample_results:
            flags = []
            
            # Check for quality_flags attribute
            if hasattr(result, 'quality_flags') and result.quality_flags:
                flags.extend(result.quality_flags)
            
            # Check for validation_flags attribute
            if hasattr(result, 'validation_flags') and result.validation_flags:
                flags.extend(result.validation_flags)
            
            if flags:
                quality_flags[result.sample_name] = flags
        
        return quality_flags
    
    def get_dataset_summary(self, dataset: XASDataset) -> Dict[str, Any]:
        """
        Generate summary statistics for a dataset.
        
        Args:
            dataset: XASDataset object
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'dataset_id': dataset.dataset_id,
            'n_samples': dataset.n_samples,
            'n_features': dataset.n_features,
            'creation_timestamp': dataset.creation_timestamp.isoformat(),
            'feature_names': dataset.feature_names,
            'sample_names': dataset.sample_names,
        }
        
        # Feature statistics
        if dataset.feature_matrix is not None:
            feature_stats = {}
            for i, name in enumerate(dataset.feature_names):
                feature_values = dataset.feature_matrix[:, i]
                feature_stats[name] = {
                    'mean': float(np.mean(feature_values)),
                    'std': float(np.std(feature_values)),
                    'min': float(np.min(feature_values)),
                    'max': float(np.max(feature_values)),
                    'n_zeros': int(np.sum(feature_values == 0)),
                    'n_valid': int(np.sum(~np.isnan(feature_values)))
                }
            summary['feature_statistics'] = feature_stats
        
        # Quality flags summary
        if dataset.quality_flags:
            total_flags = sum(len(flags) for flags in dataset.quality_flags.values())
            summary['quality_flags_summary'] = {
                'n_samples_with_flags': len(dataset.quality_flags),
                'total_flags': total_flags
            }
        
        return summary


# =============================================================================
# Standalone utility functions
# =============================================================================

def assemble_dataset_from_results(
    sample_results: List[XASSampleResult],
    metadata: Optional[Dict[str, List[Any]]] = None,
    config_path: Optional[Path] = None,
    dataset_id: Optional[str] = None
) -> XASDataset:
    """
    Convenience function for assembling a dataset.
    
    Args:
        sample_results: List of XASSampleResult objects
        metadata: Optional metadata dictionary
        config_path: Optional path to config file
        dataset_id: Optional dataset identifier
        
    Returns:
        XASDataset object
    """
    assembler = XASBatchAssembler(config_path)
    return assembler.assemble_dataset(sample_results, metadata, dataset_id)


def save_dataset_to_json(dataset: XASDataset, output_path: Path) -> None:
    """
    Save dataset to JSON file.
    
    Args:
        dataset: XASDataset object
        output_path: Path to output JSON file
    """
    # Use model_dump_json which properly handles numpy arrays via json_encoders
    with open(output_path, 'w') as f:
        f.write(dataset.model_dump_json(indent=2))
    
    logger.info(f"Dataset saved to {output_path}")


def load_dataset_from_json(input_path: Path) -> XASDataset:
    """
    Load dataset from JSON file.
    
    Args:
        input_path: Path to input JSON file
        
    Returns:
        XASDataset object
    """
    import json
    
    with open(input_path, 'r') as f:
        dataset_dict = json.load(f)
    
    # Convert lists back to numpy arrays
    if 'feature_matrix' in dataset_dict and dataset_dict['feature_matrix'] is not None:
        dataset_dict['feature_matrix'] = np.array(dataset_dict['feature_matrix'])
    
    dataset = XASDataset(**dataset_dict)
    
    logger.info(f"Dataset loaded from {input_path}")
    return dataset
